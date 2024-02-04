import os
import torch.nn.functional as F
from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 
from inference_utils.load_inference_data import *
from inference_utils.inference_class import *


current_file = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file)
checkpoint_path = current_directory + '/weights/mobilenetv2classify.pth'

# 把这个地址换成自己要推理的视频帧所在文件夹的地址
annot_path = current_directory + "/data/EgoGesture_annotation" 

inference = Inference(checkpoint_path_classify=checkpoint_path,
                      checkpoint_path_dectet=checkpoint_path,
                      annot_path=annot_path)

result = inference.inference()

