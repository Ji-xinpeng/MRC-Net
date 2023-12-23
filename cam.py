from deals.load_dataset import *
from others.train import *
from torchsummary import summary
from models.models import *
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torch.nn import functional as F
from torchcam.methods import SmoothGradCAMpp

# 加载模型
def load_model():
    print("load model")
    model = TSN(params['num_classes'], args.clip_len, 'RGB',
                        is_shift=args.is_shift,
                        partial_bn=args.npb,
                        base_model=args.base_model,
                        shift_div=args.shift_div,
                        dropout=args.dropout,
                        img_feature_dim=224,
                        pretrain='False',  # 'imagenet' or False
                        consensus_type='avg',
                        fc_lr5=True)
    
    checkpoint_path = "EgoGesture-mobilenetv2/2023-12-23-21-21-29/clip_len_8frame_sample_rate_1_checkpoint.pth.tar"
    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    print("load checkpoint {}".format(checkpoint_path))
    model.load_state_dict(pretrained_dict['state_dict'])
    return model




def main():
    model = load_model()
    cam_extractor = SmoothGradCAMpp(model)




if __name__ == "__main__":
    main()