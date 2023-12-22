import os 
import sys
import pickle
import numpy as np
import pandas as pd
import random
import torch
import pdb
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import skimage.util as ski_util
from sklearn.utils import shuffle
from copy import copy
from others.params import *


def convertLabel(labels, annot_path, mode):
    label_df = pd.read_csv(os.path.join(annot_path, '{}.csv'.format("jester-v1-labels")), header=None)
    index_to_label_dict = {}
    label_to_index_dict = {}
    index = 0
    for frame_i in range(label_df.shape[0]):
        index_to_label_dict[index] = label_df[0].iloc[frame_i].split(';')[0]
        label_to_index_dict[label_df[0].iloc[frame_i].split(';')[0]] = index
        index += 1
    new_label = []
    for l in labels:
        for k, v in label_to_index_dict.items():
            if l == k:
                new_label.append(v)
    return new_label


def load_video(annot_path, mode):  # mode 是train
    annot_df = pd.read_csv(os.path.join(annot_path, '{}.csv'.format(mode)), header=None)

    save_file = os.path.join(annot_path, '{}.pkl'.format(mode))
    annot_df.to_pickle(save_file)
    # mode: train, val, test
    csv_file = os.path.join(annot_path, '{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    rgb_samples = []
    depth_samples = []
    labels = []
    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df[0].iloc[frame_i].split(';')[0] # convert string in dataframe to list
        rgb_samples.append(rgb_list)
        label = annot_df[0].iloc[frame_i].split(';')[1]
        labels.append(label)
    print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
    labels = convertLabel(labels, annot_path, mode)
    return rgb_samples, labels


class dataset_video(Dataset):
    def __init__(self, root_path, mode, spatial_transform=None, temporal_transform=None):
        self.root_path = root_path
        self.rgb_samples, self.labels = load_video(root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        path = '../../jixinpeng/Jester/frames/' + str(rgb_name) + '/'
        rgb_p = os.listdir(path)
        rgb_p.sort()
        rgb = []
        for index in rgb_p:
            rgb.append(path+index)
        rgb_name = rgb
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_frames.append(rgb_cache)
        clip_frames = self.spatial_transform(clip_frames)
        n, h, w = clip_frames.size()
        return clip_frames.view(-1, 3, h, w), int(label)
    def __len__(self):
        return int(self.sample_num)





class dataset_video_inference(Dataset):
    def __init__(self, root_path, mode, clip_num = 2, spatial_transform=None, temporal_transform=None):
        self.root_path = root_path
        self.clip_num = clip_num
        self.video_samples, self.labels = load_video(root_path, mode)
        self.mode = mode
        self.sample_num = len(self.video_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, idx):
        rgb_name = self.video_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        video_clip = []
        for win_i in range(self.clip_num):
            clip_frames = []
            selected_indice = self.temporal_transform(copy(indices))
            for frame_name_i in selected_indice:
                rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
                clip_frames.append(rgb_cache)
            clip_frames = self.spatial_transform(clip_frames)
            n, h, w = clip_frames.size()
            video_clip.append(clip_frames.view(-1, 3, h, w)) 
        video_clip = torch.stack(video_clip)
        return video_clip, int(label)

    def __len__(self):
        return int(self.sample_num)


