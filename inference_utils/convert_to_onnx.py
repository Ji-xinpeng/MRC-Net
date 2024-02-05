import os
import sys
import onnx
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
import torch.onnx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# 获取当前文件所在目录的上两级目录的路径
two_up = Path(__file__).resolve().parents[1]
sys.path.append(str(two_up))

from inference_utils.load_inference_data import *



class ConvertToONNX:
    def __init__(self) -> None:
        self.x = None    
        self.model = None 

    def _get_onnx_save_path(self, pth_path):
        print("-------------------- pth_path : ", pth_path)
        onnx_path = pth_path.split('.')[0] + ".onnx"
        print("-------------------- onnx_path : ", onnx_path)
        return onnx_path
    
    def _check_onnx_is_succesful(self, pth_path):
        onnx_model = onnx.load(self._get_onnx_save_path(pth_path))
        onnx.checker.check_model(onnx_model)
        print("onnx has been saved to : ", self._get_onnx_save_path(pth_path))

    def convert_to_onnx(self, pth_path: str, shape: tuple):
        self.model = torch.load(pth_path, map_location=torch.device("cpu"))
        self.x = torch.randn(shape)
        with torch.no_grad():
            torch.onnx.export(
                self.model.module,
                self.x,
                self._get_onnx_save_path(pth_path),
                opset_version = 12,
                input_names = ['input'],
                output_names = ['output'],
                verbose='True'
            )
        self._check_onnx_is_succesful(pth_path)
