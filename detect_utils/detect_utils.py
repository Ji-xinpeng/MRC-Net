import os 
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import pdb
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
# 获取当前文件所在目录的上两级目录的路径
two_up = Path(__file__).resolve().parents[1]
sys.path.append(str(two_up))

from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 
from inference_utils.load_inference_data import *

def get_pkl_path():
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    pkl_path = os.path.dirname(current_directory) + '/data/EgoGesture_annotation/'
    return pkl_path


def define_spatial_transform():
    input_mean = [.485, .456, .406]
    input_std = [.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)

    scales = [1, .875, .75, .66]

    trans_train = torchvision.transforms.Compose([
        GroupScale([224, 224]),
        GroupMultiScaleCrop([224, 224], scales),
        Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
        normalize])

    trans_test = torchvision.transforms.Compose([
        GroupScale([224, 224]),
        Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
        normalize])
    
    return trans_train, trans_test


def load_detect_video(annot_path, mode):
    # mode: detect_train, detect_val
    csv_file = os.path.join(annot_path, '{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    rgb_samples = []
    labels = []

    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df['detect'].iloc[frame_i] # convert string in dataframe to list
        rgb_samples.append(rgb_list)
        labels.append(annot_df['label'].iloc[frame_i])

    print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples, labels


    
class Dataset_detect_video(Dataset):
    def __init__(self, root_path, mode, spatial_transform):
        self.root_path = root_path
        self.rgb_samples, self.labels = load_detect_video(root_path, mode)
        rgb_test = Image.open(self.rgb_samples[0]).convert("RGB")
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.__getitem__(0)


    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        label = self.labels[idx]

        rgb_cache = Image.open(rgb_name).convert("RGB")
        clip_rgb_frames = self.spatial_transform([rgb_cache])
        n, h, w = clip_rgb_frames.size()
        copy_frames = clip_rgb_frames.view(3, h, w)
        return copy_frames, copy_frames, int(label)

    def __len__(self):
        return int(self.sample_num)


def get_detect_dataloader():
    trans_train, trans_test = define_spatial_transform()
    train_detect_dataset = Dataset_detect_video(get_pkl_path(), "detect_train", trans_train)
    train_detect_dataloader = DataLoader(train_detect_dataset, batch_size=params['batch_size'], shuffle=True,
                                    num_workers=params['num_workers'])

    val_detect_dataset = Dataset_detect_video(get_pkl_path(), "detect_val", trans_test)
    val_detect_dataloader = DataLoader(val_detect_dataset, batch_size=params['batch_size'], num_workers=params['num_workers'])

    return train_detect_dataloader, val_detect_dataloader



def get_detect_model_policies(model):
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    lr5_weight = []
    lr10_bias = []
    bn = []
    custom_weight = []
    custom_bn = []

    conv_cnt = 0
    bn_cnt = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            lr5_weight.append(ps[0])
            if len(ps) == 2:
                lr10_bias.append(ps[1])
        elif isinstance(m, torch.nn.BatchNorm2d):
            bn_cnt += 1
        elif isinstance(m, torch.nn.BatchNorm1d):
            bn_cnt += 1
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
    return [
        {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
            'name': "first_conv_bias"},
        {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': "normal_weight"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
            'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
            'name': "BN scale/shift"},
        {'params': custom_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': "custom_weight"},
        {'params': custom_bn, 'lr_mult': 1, 'decay_mult': 0,
            'name': "custom_bn"},
        # for fc
        {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
            'name': "lr5_weight"},
        {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
            'name': "lr10_bias"},
    ]