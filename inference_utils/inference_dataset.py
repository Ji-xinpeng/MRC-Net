import os
import torch.nn.functional as F
from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 

def load_dataloader(annot_path):
    input_mean = [.485, .456, .406]
    input_std = [.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)

    scales = [1, .875, .75, .66]

    trans_test = torchvision.transforms.Compose([
        GroupScale([224, 224]),
        Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
        normalize])

    temporal_transform_test = torchvision.transforms.Compose([
        TemporalUniformCrop_val(args.clip_len)
    ])

    val_dataloader = DataLoader(dataset_EgoGesture.dataset_video_inference(annot_path, 'val', clip_num=1, 
                                                                            spatial_transform=trans_test, 
                                                                            temporal_transform = temporal_transform_test),
                                batch_size=1, 
                                num_workers=2)
    return val_dataloader


def main():
    checkpoint_path = '/root/autodl-tmp/MRC-Net/EgoGesture-mobilenetv2/2024-02-02-13-59-26/mobilenetv2classify.pth'
    annot_path = "/root/autodl-tmp/MRC-Net/data/EgoGesture_annotation"     

    model = torch.load(checkpoint_path) 
    model.eval()             
    
    val_dataloader = load_dataloader(annot_path) 
    with torch.no_grad():
        yes = 0
        no = 0
        for step, inputs in enumerate(tqdm(val_dataloader)):
            rgb, depth, labels = inputs[0], inputs[1], inputs[2]
            # outputs = model(val_dataloader[0])
            nb, t, n_clip, nc, h, w = rgb.size()
            rgb = rgb.view(1, 8, nc, h, w) # n_clip * nb (1) * crops, T, C, H, W
            outputs = model(rgb)
            outputs = outputs.view(nb, 1, -1)
            outputs = F.softmax(outputs, 2)
            # 找到概率最大值的索引，这就是预测值
            _, predicted_class = torch.max(outputs, 2)
            # print(f'预测的类别是: {predicted_class[0][0]}    label : {labels}')
            if int(predicted_class[0][0]) == int(labels[0]):
                yes += 1
            else:
                no += 1
        print(yes, no)



    

if __name__ == '__main__':
    main()