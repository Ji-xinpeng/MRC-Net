import os
import torch.nn.functional as F
import sys
from pathlib import Path

# 获取当前文件所在目录的上两级目录的路径
two_up = Path(__file__).resolve().parents[1]
sys.path.append(str(two_up))

from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 
from inference_utils.load_inference_data import *


class Dectect:
    def __init__(self, checkpoint_path_dectet: str,
                 annot_path: str) -> None:
        self.model_detect = torch.load(checkpoint_path_dectet)
        self.frames = []
        self.annot_path = annot_path

    def _load_dataloader(self):
        input_mean = [.485, .456, .406]
        input_std = [.229, .224, .225]
        normalize = GroupNormalize(input_mean, input_std)
        trans_test = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
            normalize])

        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])

        inference_data = inference_data_video(self.annot_path, spatial_transform=trans_test, clip_num = 1,
                                                        temporal_transform=temporal_transform_test)

        return inference_data.get_inference_data()
    

    def detect(self, image):
        pass
