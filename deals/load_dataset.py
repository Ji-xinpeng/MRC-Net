from others.params import *
from data import dataset_jester, dataset_EgoGesture, dataset_sthv2
import torch
import torchvision
from models.spatial_transforms import *
from models.temporal_transforms import *
import time
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms

# 获取当前文件路径
current_file = os.path.abspath(__file__)
# 获取所在目录路径
current_directory = os.path.dirname(current_file)
# 获取上一级目录路径
parent_directory = os.path.dirname(current_directory)

annot_path = parent_directory + '/data/{}_annotation'.format(args.dataset)
label_path = os.path.dirname(parent_directory) + '/{}/'.format(args.dataset) # for submitting testing results


def load_dataset():
    seed = 1
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    input_mean = [.485, .456, .406]
    input_std = [.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)

    scales = [1, .875, .75, .66]
    if args.dataset == 'sthv2':
        trans_train = torchvision.transforms.Compose([
            GroupMultiScaleCrop(224, scales),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
            normalize
        ])
        temporal_transform_train = torchvision.transforms.Compose([
            TemporalUniformCrop_train(args.clip_len)
        ])
        trans_test = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
            normalize
        ])
        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])
    elif args.dataset == 'jester':
        trans_train = torchvision.transforms.Compose([
            GroupScale(256),
            GroupMultiScaleCrop(224, scales),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
            normalize
        ])
        temporal_transform_train = torchvision.transforms.Compose([
            TemporalUniformCrop_train(args.clip_len)
        ])
        trans_test = torchvision.transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
            normalize
        ])
        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])

    elif args.dataset == 'EgoGesture':
        trans_train = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            GroupMultiScaleCrop([224, 224], scales),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
            normalize])

        temporal_transform_train = torchvision.transforms.Compose([
            TemporalUniformCrop_train(args.clip_len)
        ])

        trans_test = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
            normalize])

        temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])


    if args.is_train:
        cudnn.benchmark = True
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        print("Loading dataset")
        if args.dataset == 'EgoGesture':

            pkl_name_train = 'train_plus_valforsystem' if args.is_train_for_system else 'train_plus_val'

            train_dataset = dataset_EgoGesture.dataset_video(annot_path, pkl_name_train,
                                                             spatial_transform=trans_train,
                                                             temporal_transform=temporal_transform_train)

            train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                          num_workers=params['num_workers'])

            pkl_name_test = 'testforsystem' if args.is_train_for_system else 'test'

            val_dataset = dataset_EgoGesture.dataset_video(annot_path, pkl_name_test, spatial_transform=trans_test,
                                                           temporal_transform=temporal_transform_test)
            val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=params['num_workers'])
        elif args.dataset == 'jester':
            train_dataset = dataset_jester.dataset_video(annot_path, 'jester-v1-train',
                                                         spatial_transform=trans_train,
                                                         temporal_transform=temporal_transform_train)
            train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                          num_workers=params['num_workers'])
            val_dataloader = DataLoader(dataset_jester.dataset_video(annot_path, 'jester-v1-validation',
                                                                     spatial_transform=trans_test,
                                                                     temporal_transform=temporal_transform_test),
                                        batch_size=params['batch_size'], num_workers=params['num_workers'])
        elif args.dataset == 'sthv2':
            train_dataset = dataset_sthv2.dataset_video(annot_path, 'train', spatial_transform=trans_train,
                                                        temporal_transform=temporal_transform_train)
            train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                          num_workers=params['num_workers'])
            val_dataset = dataset_sthv2.dataset_video(annot_path, 'val', spatial_transform=trans_test,
                                                      temporal_transform=temporal_transform_test)
            val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=params['num_workers'])

    return train_dataloader, val_dataloader

