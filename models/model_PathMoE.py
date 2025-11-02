import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import csv  
import math
"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = True, n_classes = 2):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 2):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.attention = nn.Linear(512*4, 4)
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        self.k_sample = min(self.k_sample, A.size(1))
        
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h1, h2, h3, h4, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h1.device

        feat = [h1, h2, h3, h4]
        feat_M = []



        

        for i in range(4):
            h = feat[i]
            A, h = self.attention_net(h)  # NxK        
            A = torch.transpose(A, 1, 0)  # KxN
            if attention_only:
                return A
            A_raw = A
            A = F.softmax(A, dim=1)  # softmax over N

            if instance_eval:
                total_inst_loss = 0.0
                all_preds = []
                all_targets = []
                inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item()
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1: #in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss

                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)

            feat_M[i] = torch.mm(A, h)

        h1 = feat_M[0]
        h2 = feat_M[1]
        h3 = feat_M[2]
        h4 = feat_M[3]

        combined_features = torch.cat([h1, h2, h3, h4], dim=1)
        attention_scores = self.attention(combined_features)
        attention_weights = F.softmax(attention_scores, dim=1)

        # 使用注意力权重加权各自特征
        h1_features = attention_weights[:, 0].unsqueeze(1) * h1
        h2_features = attention_weights[:, 1].unsqueeze(1) * h2
        h3_features = attention_weights[:, 2].unsqueeze(1) * h3
        h4_features = attention_weights[:, 3].unsqueeze(1) * h4


        # 将加权特征相加得到最终融合特征
        M = h1_features + h2_features + h3_features + h4_features

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1] # 预测标签
        Y_prob = F.softmax(logits, dim = 1) # 预测概率
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
class ReduceLengthByQuarter(nn.Module):
    def __init__(self):
        super(ReduceLengthByQuarter, self).__init__()

    def forward(self, x):
        batch_size, seq_len = x.size()
        new_seq_len = seq_len // 4
        linear = nn.Linear(seq_len, new_seq_len).to(x.device)
        x = linear(x)
        return x
class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
class Gate(nn.Module):
    def __init__(self, input_size, num_experts):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_size, num_experts)
    
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)   

class PathMoE(CLAM):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, num_experts = 2):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256, 128], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        self.attention = nn.Linear(512*5, 5)
        # self.multi_head_attention = MultiHeadAttention(512*5, 5)
        self.fc = nn.Linear(5 * 5, 5)
        nn.init.xavier_uniform_(self.fc.weight)

        fc = [nn.Linear(size[1], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.encoder1 = nn.Sequential(nn.Dropout(0.25), nn.Linear(size[1], size[1]), nn.ReLU(),
                                      nn.Dropout(0.25),
                                      nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25))
        self.encoder2 = nn.Sequential(nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25))
        self.proj_list = nn.ModuleList([nn.Linear(2560, 512) for _ in range(5)])
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.fc = nn.Linear(size[1],size[1])
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.reduce_layer = ReduceLengthByQuarter()
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.training = True

        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(512, 1024, 512) for _ in range(num_experts)])
        self.gate = Gate(512, num_experts)


        initialize_weights(self)

    def forward(self, h1, h2, h3, h4, h5, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h1.device

     
        

        feat = [h1, h2, h3, h4, h5]

        feat_M = [None] * 5
        total_inst_loss = [0.0] * 5


        # exit
        all_preds = []
        all_targets = []
        all_gates = []
        group_hard = [[] for _ in range(5)] 

        for k in range(5):
            h = feat[k].to(device)
            h = self.proj_list[k](h)

            A, h = self.attention_net(h)  # NxK        
            A = torch.transpose(A, 1, 0)  # KxN
            if attention_only:
                return A
            A_raw = A
            A = F.softmax(A, dim=1)  # softmax over N


            if instance_eval:
                
                all_preds = []
                all_targets = []
                inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item()
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1: #in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss[k] += instance_loss

                if self.subtyping:
                    total_inst_loss[k] /= len(self.instance_classifiers)

            MM = torch.mm(A, h).to(device)
            
            gate_outputs = self.gate(MM)
            # print("gate score:", gate_outputs)
            hard_gate_outputs = self.hard_gate(gate_outputs)
            group_hard[k].append(hard_gate_outputs.detach())  # 
            expert_outputs = torch.stack([expert(MM) for expert in self.experts], dim=1)
            moe_output = torch.sum(hard_gate_outputs.unsqueeze(2) * expert_outputs, dim=1)
            all_gates.append(gate_outputs)
            feat_M[k] = moe_output

        hm1 = feat_M[0]      
        hm2 = feat_M[1]
        hm3 = feat_M[2]
        hm4 = feat_M[3]
        hm5 = feat_M[4]


        combined_features = torch.cat([hm1, hm2, hm3, hm4, hm5], dim=1)
        combined_features = combined_features / combined_features.norm(dim=1, keepdim=True)  # 归一化处理

        attention_output = self.attention(combined_features)


        attention_weights = F.softmax(attention_output, dim=1)



        h1_features = attention_weights[:, 0].unsqueeze(1) * hm1
        h2_features = attention_weights[:, 1].unsqueeze(1) * hm2
        h3_features = attention_weights[:, 2].unsqueeze(1) * hm3
        h4_features = attention_weights[:, 3].unsqueeze(1) * hm4
        h5_features = attention_weights[:, 4].unsqueeze(1) * hm5


        fusion = h1_features + h2_features + h3_features + h4_features + h5_features

        M = self.encoder1(fusion)


        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        Y_prob = F.softmax(logits, dim = 1)


       
        

        losses = []
        for g in range(5):
            if len(group_hard[g]) == 0:
                continue
            hard_cat = torch.cat(group_hard[g], dim=0).float()  # [K_g, 2]
            frac = hard_cat.mean(dim=0)                         # [2], 
            losses.append(((frac - 0.5)**2).mean()) 



        if len(losses) > 0:
            balance_loss = torch.stack(losses).mean()
        else:
            balance_loss = torch.tensor(0.0, device=device)

        if instance_eval:
            results_dict = {'balance_loss': balance_loss,'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds), 'gates':  np.array(all_gates)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
    

    def hard_gate(self, gate_outputs):
        max_indices = torch.argmax(gate_outputs, dim=1, keepdim=True)
        hard_gate_outputs = torch.zeros_like(gate_outputs).scatter_(1, max_indices, 1)
        return hard_gate_outputs
    
