import os
import glob
import subprocess
import torch.nn.functional as F
from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 
from inference_utils.load_inference_data import *
from inference_utils.inference_class import *
from inference_utils.convert_to_onnx import *
from data.dataset_EgoGesture import * 


current_file = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file)
checkpoint_path_classify = current_directory + '/weights/mobilenetv2classify.pth'
checkpoint_path_detect = current_directory + '/weights/mobilenetv2detect.pth'

# 把模型转换成 onnx 格式
# 使用glob模块匹配文件夹下以.onnx结尾的文件
extension = ".onnx"
files = glob.glob(os.path.join(current_directory + '/weights/', f"*{extension}"))
if not files:
    convert = ConvertToONNX()
    convert.convert_to_onnx(checkpoint_path_classify, (1, 8, 3, 224, 224))
    convert.convert_to_onnx(checkpoint_path_detect, (1, 3, 224, 224))

# 把这个地址换成自己要推理的视频帧所在文件夹的地址
annot_path = current_directory + "/data/EgoGesture_annotation" 

inference = Inference(checkpoint_path_classify = checkpoint_path_classify, 
                      checkpoint_path_detect = checkpoint_path_detect)

# inference.load_dataloader()
# result = inference.inference_pth()
# result = inference.inference_onnx()
# image = Image.open('C:/Users/ji/Desktop/MRC-Net/test_data/u=325784059,3802711583&fm=253&fmt=auto&app=138&f=JPEG.webp').convert("RGB")
# detect_result = inference.inference_detect_pth(image)

inference.capture_run()
