import os
import subprocess
import torch.nn.functional as F
from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 
from inference_utils.load_inference_data import *
from inference_utils.inference_class import *
from inference_utils.convert_to_onnx import *


current_file = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file)
checkpoint_path_classify = current_directory + '/weights/mobilenetv2classify.pth'
checkpoint_path_detect = current_directory + '/weights/mobilenetv2detect.pth'

# 把模型转换成 onnx 格式
convert = ConvertToONNX()
convert.convert_to_onnx(checkpoint_path_classify, (1, 8, 3, 224, 224))
convert.convert_to_onnx(checkpoint_path_detect, (1, 3, 224, 224))

# 把这个地址换成自己要推理的视频帧所在文件夹的地址
annot_path = current_directory + "/data/EgoGesture_annotation" 

inference = Inference(checkpoint_path_classify = checkpoint_path_classify, 
                      checkpoint_path_detect = checkpoint_path_detect,
                      annot_path = annot_path)

inference.load_dataloader()
result = inference.inference_pth()
result = inference.inference_onnx()

# command = "trtexec --onnx=/root/autodl-tmp/MRC-Net/weights/mobilenetv2classify.onnx \
#           --saveEngine=/root/autodl-tmp/MRC-Net/weights/mobilenetv2classify.trt --explicitBatch"
# t = subprocess.run(command, shell=True, capture_output=True, text=True)
# print(t.stdout)