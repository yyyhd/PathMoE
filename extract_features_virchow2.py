import glob

import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide

import timm
import torchvision
from torchvision import transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def virchow2_vit_h14():
    timm_kwargs = {
        "img_size": 224,
        "init_values": 1e-5,
        "num_classes": 0,
        "reg_tokens": 4,
        "mlp_ratio": 5.3375,
        "global_pool": "",
        "dynamic_img_size": True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
                }
    model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
  

    
    return model
def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, target_patch_size=-1):
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    x, y = dataset[0]
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    i = 0
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))

            batch = batch.to(device, non_blocking=True)

            batch_transformed = []
            for img in batch:
                img = transforms.ToPILImage()(img.cpu())
                img = virchow2_transforms(img)
                batch_transformed.append(img)
            batch = torch.stack(batch_transformed).to(device)

            output = model(batch)  # size: B x 261 x 1280
            class_token = output[:, 0]                   # B x 1280
            patch_tokens = output[:, 5:]                 # B x 256 x 1280
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # B x 2560

            features = embedding.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='/home/hoo/projects/zzhuo/result/patch/benyuan_another_presets2')
parser.add_argument('--data_slide_dir', type=str, default='/home/hoo/projects/zzhuo/WSI/duoranse2')
# parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--slide_ext', type=str, default='.mrxs')
parser.add_argument('--csv_path', type=str,
                    default='/home/hoo/projects/zzhuo/result/patch/benyuan_another_presets2/process_list_autogen.csv')

parser.add_argument('--feat_dir', type=str, default='/home/hoo/projects/zzhuo/result/features/virchow2')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--virchow2_weight', type=str, default='/home/hoo/projects/zzhuo/CLAM_master/pre-train/virchow2/pytorch_model.bin')
args = parser.parse_args()

if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    print('loading model checkpoint')
    model = virchow2_vit_h14()
    model.load_state_dict(torch.load(args.virchow2_weight, map_location="cpu"), strict=True)
    
    model = model.to(device)
    virchow2_transforms = transforms.Compose(
            [
                transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    total = len(bags_dataset)
    print(total)


    for bag_candidate_idx in range(total):
        slide_dir = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        slide_id = os.path.basename(os.path.dirname(bags_dataset[bag_candidate_idx]))
        print(slide_id)

        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        patient_id = slide_id.split("-")[0]
        slide_file_dir = os.path.join(args.data_slide_dir, patient_id)
        # print(h5_file_path)
        print(slide_file_dir)
        
        if not os.path.exists(slide_file_dir):
            print('1111111')
            continue
        if not os.path.exists(h5_file_path):
            print('22222')
            continue
            continue
        slide_file_path = glob.glob(os.path.join(slide_file_dir, slide_id,'*.mrxs'))[0]#IHC
        # slide_file_path =  bags_dataset[bag_candidate_idx]
        # slide_file_path = glob.glob(os.path.join(slide_file_dir,'*.mrxs'))[0]#HE
        print("666",slide_file_path)

        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)
        #
        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                            model=model, batch_size=args.batch_size,
                                            verbose=1, print_every=20,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

        file = h5py.File(output_file_path, "r")
        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))



