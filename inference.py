import os
import torch.nn.functional as F
from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 
from inference_utils.utils import *



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

    inference_data = inference_data_video(annot_path, spatial_transform=trans_test, clip_num = 1,
                                                    temporal_transform=temporal_transform_test)

    return inference_data.get_inference_data()


def main():
    checkpoint_path = '/root/autodl-tmp/MRC-Net/EgoGesture-mobilenetv2/2024-02-02-15-00-40/mobilenetv2classify.pth'

    # 把这个地址换成自己要推理的视频帧所在文件夹的地址
    annot_path = "/root/autodl-tmp/MRC-Net/data/EgoGesture_annotation"     

    model = torch.load(checkpoint_path) 
    model.eval()             
    
    # 加载处理自己的数据
    val_dataloader = load_dataloader(annot_path) 

    with torch.no_grad():
        outputs = model(val_dataloader[0])
        outputs = outputs.view(1, 1, -1)
        outputs = F.softmax(outputs, 2)
        # 找到概率最大值的索引，这就是预测值
        _, predicted_class = torch.max(outputs, 2)
        print(f'预测的类别是: {predicted_class[0][0]}')


if __name__ == '__main__':
    main()