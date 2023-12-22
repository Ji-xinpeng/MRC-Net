from others.params import *
from data import dataset_jester, dataset_EgoGesture, dataset_sthv2
import torch
import torchvision
from models.spatial_transforms import *
from models.temporal_transforms import *
import time
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from PIL import Image
import torchvision
import torchvision.transforms as transforms


annot_path = './data/{}_annotation'.format(args.dataset)
label_path = '../{}/'.format(args.dataset) # for submitting testing results


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

            train_dataset = dataset_EgoGesture.dataset_video(annot_path, 'train_plus_val',
                                                             spatial_transform=trans_train,
                                                             temporal_transform=temporal_transform_train)

            train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                          num_workers=params['num_workers'])
            val_dataset = dataset_EgoGesture.dataset_video(annot_path, 'test', spatial_transform=trans_test,
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

