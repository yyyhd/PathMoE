import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits

from models.model_PathMoE import PathMoE, CLAM

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from pandas import DataFrame
import pandas as pd

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=25, stop_epoch=40, verbose=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """
        train for a single fold
    """

    print('\nTraining Fold {}!'.format(cur))

    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam', 'PathMoE']:  

        if args.B > 0:  
            model_dict.update({'k_sample': args.B})

        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type == 'clam':
            model = CLAM(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'PathMoE':
            model = PathMoE(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

  

    model.relocate()  
    model = model.to(device)
    print('Done!')
    print_network(model)


    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')

    train_loader = get_split_loader1(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)

    
    val_loader = get_split_loader1(val_split, testing=args.testing)

    test_loader = get_split_loader1(test_split, testing=args.testing)

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30, stop_epoch=60, verbose=True)

    else:
        early_stopping = None
    print('Done!')
    i = 0

    for epoch in range(args.max_epochs):

        i = i + 1
        train_name_list = []
        train_score = []
        
        train_lable = []
        val_name_list = []
        val_score = []

        val_lable = []
        
        if args.model_type in ['clam', 'PathMoE'] and not args.no_inst_cluster:
            train_name, train_score, train_lables = train_loop(i, train_name_list, train_score,
                                                                     train_lable, epoch,
                                                                     model, train_loader, optimizer,
                                                                     args.n_classes, args.bag_weight,
                                                                     writer, loss_fn)
            train_name = DataFrame(train_name)
            train_name.columns = ['id']
            train_lables = DataFrame(train_lables)
            train_lables.columns = ['lable']
            train_score = DataFrame(train_score)
            train_score.columns = ['prob']
            

            list = pd.concat([train_name, train_lables], axis=1, join='inner')
            csv = DataFrame(list)
            
            csv.set_index(['id'], inplace=True)
            csv.to_csv(os.path.join(args.result_dir,'train_result.csv'))
            stop, val_name, val_score, val_lables = validate_(val_name_list, val_score, 
                                                                     val_lable, cur, epoch, model,
                                                                     val_loader, args.n_classes,
                                                                     early_stopping, writer, loss_fn,
                                                                     args.results_dir)
            val_name = DataFrame(val_name)
            val_name.columns = ['id']
            val_lables = DataFrame(val_lables)
            val_lables.columns = ['label']
            val_score = DataFrame(val_score)
            val_score.columns = ['prob']
            
            v_list = pd.concat([val_name, val_lables], axis=1, join='inner')
            v_csv = DataFrame(v_list)
            
            v_csv.set_index(['id'], inplace=True)
            v_csv.to_csv(os.path.join(args.result_dir,'val_result.csv'))


        else:
            train_name, train_score, train_lables = train_loop1(i, train_name_list, train_score,
                                                                                 train_lable, epoch, model,
                                                                                train_loader, optimizer, args.n_classes,
                                                                                writer, loss_fn)
            train_name = DataFrame(train_name)
            train_name.columns = ['id']
            train_lables = DataFrame(train_lables)
            train_lables.columns = ['lable']
            train_score = DataFrame(train_score)
            train_score.columns = ['prob']
            
            list = pd.concat([train_name, train_lables], axis=1, join='inner')
            csv = DataFrame(list)

            
            csv.set_index(['id'], inplace=True)
            csv.to_csv(os.path.join(args.result_dir,'train_result.csv'))

            stop, val_name, val_score, val_lables = validate(val_name_list, val_score,
                                                                            val_lable, cur, epoch, model, val_loader,
                                                                            args.n_classes,
                                                                            early_stopping, writer, loss_fn,
                                                                            args.results_dir)
            val_name = DataFrame(val_name)
            val_name.columns = ['id']
            val_lables = DataFrame(val_lables)
            val_lables.columns = ['label']
            val_score = DataFrame(val_score)
            val_score.columns = ['prob']
            
            v_list0 = pd.concat([val_name, val_lables], axis=1, join='inner')
            v_csv = DataFrame(v_list0)
            
            
            v_csv.set_index(['id'], inplace=True)
            v_csv.to_csv(os.path.join(args.result_dir,'val_result.csv'))
        epoch_save_dir = os.path.join(args.results_dir, "epoch_save")
        if not os.path.exists(epoch_save_dir):
            os.makedirs(epoch_save_dir)

        torch.save(model.state_dict(), os.path.join(epoch_save_dir, f"s_{cur}_epoch_{epoch}.pt"))
        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, vname, vlabel, vscore = summary(model, val_loader, args.n_classes)

    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger, test_name, test_label, test_score = summary(model,
                                                                                                  test_loader,
                                                                                                  args.n_classes)
    test_name = DataFrame(test_name)
    test_name.columns = ['id']
    test_label = DataFrame(test_label)
    test_label.columns = ['label']
    test_score = DataFrame(test_score)
    test_score.columns = ['prob']
    
    t_list = pd.concat([test_name, test_label], axis=1, join='inner')
    t_csv = DataFrame(t_list)
    
    
    t_csv.set_index(['id'], inplace=True)
    t_csv.to_csv(os.path.join(args.result_dir,'test_result.csv'))

    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error


def train_loop(i, name_list, score_list, label_list, epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx,(data1, data2, data3, data4, data5, label, slide_id) in enumerate(loader):
     # iii
        slide_id = str(slide_id)

        name_list.append(slide_id)
        labelss = int(label)
        labelss = str(labelss)
        label_list.append(labelss)
        data1, data2,data3, data4, data5, label = data1.to(device), data2.to(device),data3.to(device),data4.to(device),data5.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, instance_dict = model(data1, data2,data3,data4,data5, label=label, instance_eval=True)
        Y_prob_list = Y_prob.tolist()[0]
        score_list.append(Y_prob_list[0])
        

        # print()
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss_total = sum(instance_dict['instance_loss'])
        instance_loss = instance_loss_total / len(instance_dict['instance_loss'])

        balance_loss = instance_dict['balance_loss']


        total_loss = bag_weight * loss + (1 - bag_weight)/2 * (instance_loss + balance_loss)
       

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('epoch {}, batch {},  weighted_loss: {:.4f}, '.format(i, batch_idx,  total_loss.item()) )


        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
    return name_list, score_list, label_list


def train_loop1(i, name_list, score_list, label_list, epoch, model, loader, optimizer, n_classes,
               writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx,(data1, data2, data3, data4, data5, label, slide_id) in enumerate(loader):

        # slide_id = int(slide_id)
        slide_id = str(slide_id)
        name_list.append(slide_id)
        labelss = int(label)
        labelss = str(labelss)
        label_list.append(labelss)

        data1, data2, data3, data4,data5, label = data1.to(device), data2.to(device), data3.to(device), data4.to(device),data5.to(device), label.to(device)

        logits, Y_prob, Y_hat, _ = model(data1, data2,data3, data4, data5)

        Y_prob_list = Y_prob.tolist()[0]
        score_list.append(Y_prob_list[0])
        

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('epoch {}, batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(i, batch_idx, loss_value,
                                                                                     label.item(), data1.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

    return name_list, score_list,label_list



def validate(name_list, score_list, label_list, cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data1, data2, data3, data4, data5, label, slide_id) in enumerate(loader):
            data1, data2, data3, data4, data5, label = data1.to(device, non_blocking=True), data2.to(device, non_blocking=True), data3.to(device, non_blocking=True), data4.to(device, non_blocking=True), data5.to(device, non_blocking=True),label.to(device, non_blocking=True)

            # slide_id = int(slide_id)
            slide_id = str(slide_id)
            name_list.append(slide_id)
            labelss = int(label)
            labelss = str(labelss)
            label_list.append(labelss)

            logits, Y_prob, Y_hat, _ = model(data1, data2, data3,data4,data5)

            Y_prob_list = Y_prob.tolist()[0]
            score_list.append(Y_prob_list[0])
            

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, name_list, score_list,  label_list

    return False, name_list, score_list, label_list

def validate_(name_list, score_list, label_list, cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, data4, data5, label, slide_id) in enumerate(loader):

            slide_id = str(slide_id)
            name_list.append(slide_id)
            labelss = int(label)
            labelss = str(labelss)
            label_list.append(labelss)
            data1, data2, data3, data4, data5, label = data1.to(device), data2.to(device), data3.to(device), data4.to(device),data5.to(device), label.to(device)
            logits, Y_prob, Y_hat, _,instance_dict = model(data1, data2, data3, data4, data5, label=label, instance_eval=True)
            Y_prob_list = Y_prob.tolist()[0]
            score_list.append(Y_prob_list[0])
            

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss_total = sum(instance_dict['instance_loss'])
            instance_loss = instance_loss_total / len(instance_dict['instance_loss'])
            
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value



            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, name_list, score_list,  label_list
    return False, name_list, score_list,  label_list

def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    test_name_list = []
    test_score = []
    
    test_label = []
    for batch_idx, (data1, data2,data3, data4, data5, label, slide_id1) in enumerate(loader):
        # slide_id1 = int(slide_id1)
        slide_id1 = str(slide_id1)
        test_name_list.append(slide_id1)
        labelss = int(label)
        labelss = str(labelss)
        test_label.append(labelss)
        data1, data2, data3, data4, data5, label = data1.to(device), data2.to(device), data3.to(device), data4.to(device),data5.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _ ,_= model(data1, data2, data3, data4, data5)

        Y_prob_list = Y_prob.tolist()[0]
        test_score.append(Y_prob_list[0])
        

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger, test_name_list, test_label, test_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    test_name_list = []
    test_score = []
    
    test_label = []
    for batch_idx, (data1, data2,data3, data4, data5, label, slide_id1) in enumerate(loader):
        # slide_id1 = int(slide_id1)
        slide_id1 = str(slide_id1)
        test_name_list.append(slide_id1)
        labelss = int(label)
        labelss = str(labelss)
        test_label.append(labelss)
        data1, data2, data3, data4, data5, label = data1.to(device), data2.to(device), data3.to(device), data4.to(device),data5.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _ ,_= model(data1, data2, data3, data4, data5)

        Y_prob_list = Y_prob.tolist()[0]
        test_score.append(Y_prob_list[0])
        

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger, test_name_list, test_label, test_score
