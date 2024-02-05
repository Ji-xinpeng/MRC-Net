import os 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import skimage.util as ski_util
from sklearn.utils import shuffle
from copy import copy
import torchvision.transforms.functional as TF


def load_inference_data(inference_data_path):
    images_paths = os.listdir(inference_data_path)
    rgb_samples = [inference_data_path + '/' + image for image in images_paths]
    rgb_samples = ['../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000178.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000179.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000180.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000181.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000182.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000183.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000184.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000185.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000186.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000187.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000188.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000189.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000190.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000191.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000192.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000193.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000194.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000195.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000196.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000197.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000198.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000199.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000200.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000201.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000202.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000203.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000204.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000205.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000206.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000207.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000208.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000209.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000210.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000211.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000212.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000213.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000214.jpg', '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000215.jpg', 
                   '../EgoGesture/frames/Subject01/Scene1/Color/rgb1/000216.jpg']
    return rgb_samples


class inference_data_video(Dataset):
    def __init__(self, root_path, clip_num, spatial_transform, temporal_transform):
        self.root_path = root_path
        self.rgb_samples = load_inference_data(root_path)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.clip_num = clip_num

    def get_inference_data(self):
        indices = [i for i in range(len(self.rgb_samples))]
        selected_indice = self.temporal_transform(copy(indices))
        clip_rgb_frames = []
        for frame_name_i in selected_indice:
            rgb_cache = Image.open(self.rgb_samples[frame_name_i]).convert("RGB")
            clip_rgb_frames.append(rgb_cache)
        clip_rgb_frames = self.spatial_transform(clip_rgb_frames)
        ntc, h, w = clip_rgb_frames.size()
        clip_rgb_frames = np.reshape(clip_rgb_frames, (1, -1, 3, h, w))

        return clip_rgb_frames



def detect_gestures_appear():
    '''隔 帧 检 测'''
    pass