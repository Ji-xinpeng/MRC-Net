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

class Inference:
    def __init__(self, checkpoint_path_classify: str,
                 annot_path: str) -> None:
        self.model_classify = torch.load(checkpoint_path_classify) 
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
    
    def inference(self):
        self.model_classify.cuda()
        self.model_classify.eval()
        # 加载处理自己的数据
        val_dataloader = self._load_dataloader() 
        print("模型所在设备:", next(self.model_classify.parameters()).device)
        print("张量所在设备:", val_dataloader.device)
        print(val_dataloader.shape)
        with torch.no_grad():
            outputs = self.model_classify(val_dataloader)
            outputs = outputs.view(1, 1, -1)
            outputs = F.softmax(outputs, 2)
            # 找到概率最大值的索引，这就是预测值
            _, predicted_class = torch.max(outputs, 2)
            print(f'预测的类别是: {predicted_class[0][0]}')
        return int(predicted_class[0][0])
    