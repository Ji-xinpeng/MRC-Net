import os
import torch.nn.functional as F
import sys
from pathlib import Path
import onnxruntime
import time
import numpy as np
import tensorrt as trt
 
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

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
                 checkpoint_path_detect: str,
                 annot_path: str) -> None:
        self.checkpoint_path_classify = checkpoint_path_classify
        self.checkpoint_path_detect = checkpoint_path_detect
        self.model_classify = torch.load(checkpoint_path_classify, map_location=torch.device("cpu")) 
        self.model_detect = torch.load(checkpoint_path_detect, map_location=torch.device("cpu")) 
        self.model_detect.cuda()
        self.model_detect.eval()
        self.onnxruntime = None
        self.onnxruntime_detect= None
        self.frames = []
        self.annot_path = annot_path
        self.val_dataloader = None
        self.spatial_transform = None
        self.temporal_transform_test = None
        self._get_spatial_temporal()
        self._build_onnxruntime()


    def _get_onnx_path(self, checkpoint_path):
        onnx_path = checkpoint_path.split('.')[0] + '.onnx'
        return onnx_path
    
    
    def _get_spatial_temporal(self):
        input_mean = [.485, .456, .406]
        input_std = [.229, .224, .225]
        normalize = GroupNormalize(input_mean, input_std)
        self.spatial_transform = torchvision.transforms.Compose([
            GroupScale([224, 224]),
            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
            normalize])

        self.temporal_transform_test = torchvision.transforms.Compose([
            TemporalUniformCrop_val(args.clip_len)
        ])

    
    def _preprocess_image(self, image: torch.Tensor):
        clip_rgb_frames = self.spatial_transform([image])
        nc, h, w = clip_rgb_frames.size()
        return np.reshape(clip_rgb_frames, (1, nc, h, w))


    def load_dataloader(self):
        inference_data = inference_data_video(self.annot_path, spatial_transform=self.spatial_transform, clip_num = 1,
                                                        temporal_transform=self.temporal_transform_test)
        self.val_dataloader = inference_data.get_inference_data()
    

    def inference_pth(self):
        start_time = time.perf_counter()
        self.model_classify.cuda()
        self.model_classify.eval()
        # print("模型所在设备:", next(self.model_classify.parameters()).device)
        # print("张量所在设备:", self.val_dataloader.device)
        with torch.no_grad():
            outputs = self.model_classify(self.val_dataloader)
            outputs = outputs.view(1, 1, -1)
            outputs = F.softmax(outputs, 2)
            # 找到概率最大值的索引，这就是预测值
            _, predicted_class = torch.max(outputs, 2)
            print(f'预测的类别是: {predicted_class[0][0]}')
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"pth 代码运行时间: {execution_time}秒")
        return int(predicted_class[0][0])
    
    
    def inference_detect_pth(self, image: torch.Tensor):
        start_time = time.perf_counter()
        # print("模型所在设备:", next(self.model_detect.parameters()).device)
        # print("张量所在设备:", self.val_dataloader.device)
        with torch.no_grad():
            image = self._preprocess_image(image)
            outputs = self.model_detect(image)
            outputs = outputs.view(1, -1)
            outputs = F.softmax(outputs, 1)
            # 找到概率最大值的索引，这就是预测值
            _, predicted_class = torch.max(outputs, 1)
            print(f'预测的类别是: {predicted_class[0]}')
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        # 计算代码运行时间
        print(f"pth 代码运行时间: {execution_time}秒")
        return int(predicted_class[0])
    

    def _build_onnxruntime(self):
        if 'CUDAExecutionProvider' in onnxruntime .get_available_providers():
            print("This ONNX Runtime build has CUDA support.")
            providers = ['CUDAExecutionProvider']
        else:
            print("This ONNX Runtime build does not have CUDA support.")
            providers = None
        self.onnxruntime = onnxruntime.InferenceSession(self._get_onnx_path(self.checkpoint_path_classify), providers = providers)
        self.onnxruntime_detect = onnxruntime.InferenceSession(self._get_onnx_path(self.checkpoint_path_detect), providers = providers)
    
    
    def inference_onnx(self):
        start_time = time.perf_counter()
        input_name = self.onnxruntime.get_inputs()[0].name
        outputs = self.onnxruntime.run(None, {input_name: np.array(self.val_dataloader)})
        outputs = torch.tensor(outputs).view(1, 1, -1)
        outputs = F.softmax(outputs, 2)
        _, predicted_class = torch.max(outputs, 2)
        print(f'onnx 预测的类别是: {predicted_class[0][0]}')
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"onnx 代码运行时间: {execution_time}秒")
        return int(predicted_class[0][0])
    