# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch.nn.utils import clip_grad_norm_
from CAM3D.ops.dataset import TSNDataSet
from CAM3D.ops.models import TSN
from CAM3D.ops.transforms import *
from CAM3D.opts import parser
from CAM3D.ops.utils import AverageMeter, accuracy
from CAM3D.ops.temporal_shift import make_temporal_pool
from tensorboardX import SummaryWriter
from CAM3D.ops.CAM3D_module import ClassAttentionMapping
import cv2
from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 

def load_model():
    print("load model")
    model = TSN(params['num_classes'], args.clip_len, 'RGB',
                        is_shift=args.is_shift,
                        partial_bn=args.npb,
                        base_model=args.base_model,
                        shift_div=args.shift_div,
                        dropout=args.dropout,
                        img_feature_dim=224,
                        pretrain='imagenet',  # 'imagenet' or False
                        consensus_type='avg',
                        fc_lr5=True)
    
    checkpoint_path = "EgoGesture-resnet50/2023-12-26-22-36-58/clip_len_8frame_sample_rate_1_checkpoint.pth.tar"
    # checkpoint_path = "/home/ubuntu/Desktop/jixinpeng/clip_len_8frame_sample_rate_1_checkpoint.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    try:
        model_dict = model.module.state_dict()
    except AttributeError:
        model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and 'fc' not in k}
    print("load pretrained model {}".format(checkpoint_path))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def cam_process(val_loader, model, norm_param, break_index):
    cam = ClassAttentionMapping(model, 'base_model.layer4', 'new_fc.weight', norm_param)
    i = 0
    for input, _, target in val_loader:
        if i == break_index:
            cam_out = cam(input[0].cuda())
            for i,(src_img,dst_img) in enumerate(zip(*cam_out)):
                cv2.imwrite("cam_results/srcimages/" + str(i + 1) + "_src.jpg", src_img)
                cv2.imwrite("cam_results/camimages/" + str(i + 1) + "_dst.jpg", dst_img)
            break
        i += 1


def main():
    # 加载模型                                       
    model = load_model()    
    model = model.cuda()                                                                                            

    input_mean = model.input_mean
    input_std = model.input_std
    norm_param = (input_mean, input_std)
    
    # 加载数据集
    train_loader, val_loader = load_dataset()

    break_index = 1
    cam_process(train_loader, model, norm_param, break_index)

    

if __name__ == '__main__':
    main()
