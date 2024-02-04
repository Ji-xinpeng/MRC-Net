import os
import sys
import onnx
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# 获取当前文件所在目录的上两级目录的路径
two_up = Path(__file__).resolve().parents[1]
sys.path.append(str(two_up))

from inference_utils.load_inference_data import *



class ConvertToONNX:
    def __init__(self, pth_path: str) -> None:
        self.pth_path = pth_path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.randn(1, 8, 3, 224, 224)     

    def _convert_to_onnx_prepare(self):
        self.model.eval().to(self.device)
        self.x.to(self.device)
        output = self.model(self.x)
        print("convert to onnx : ", output.shape)

    def _get_onnx_save_path(self):
        print("-------------------- pth_path : ", self.pth_path)
        onnx_path = self.pth_path.split('.')[0] + ".onnx"
        print("-------------------- onnx_path : ", onnx_path)
        return onnx_path
    
    def _check_onnx_is_succesful(self):
        onnx_model = onnx.load(self._get_onnx_save_path())
        onnx.checker.check_model(onnx_model)
        print("onnx has been saved to : ", self._get_onnx_save_path())

    def convert_to_onnx(self):
        self.model = torch.load(self.pth_path)
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                self.x,
                self._get_onnx_save_path(),
                input_names = ['input'],
                output_names = ['output']
            )
        self._check_onnx_is_succesful()


convert = ConvertToONNX('/root/autodl-tmp/MRC-Net/weights/mobilenetv2classify.pth')
convert.convert_to_onnx()