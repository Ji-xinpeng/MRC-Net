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

def get_pkl_path():
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    pkl_path = os.path.dirname(current_directory) + '/data/EgoGesture_annotation/'
    return pkl_path


def load_dectect_video(annot_path, mode):
    # mode: dectect_train, dectect_val
    csv_file = os.path.join(annot_path, '{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    print(annot_df[:5])
    rgb_samples = []
    labels = []

    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df['dectect'].iloc[frame_i] # convert string in dataframe to list
        rgb_samples.append(rgb_list)
        labels.append(annot_df['label'].iloc[frame_i])

    print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples, labels


class dataset_dectect_video(Dataset):
    def __init__(self, root_path, mode, spatial_transform, temporal_transform):
        self.root_path = root_path
        self.rgb_samples, self.labels = load_dectect_video(root_path, mode)
        rgb_test = Image.open(self.rgb_samples[0][0]).convert("RGB")
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.__getitem__(0)


    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_rgb_frames = []
        clip_depth_frames = []

        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_rgb_frames.append(rgb_cache)

        clip_rgb_frames = self.spatial_transform(clip_rgb_frames)

        n, h, w = clip_rgb_frames.size()
        copy_frames = clip_rgb_frames.view(-1, 3, h, w)

        return clip_rgb_frames.view(-1, 3, h, w), clip_depth_frames.view(-1, 1, h, w), int(label)

    def __len__(self):
        return int(self.sample_num)



load_dectect_video(get_pkl_path(), "val")