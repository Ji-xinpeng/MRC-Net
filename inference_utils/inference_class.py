import os
import torch.nn.functional as F
import sys
from pathlib import Path
import onnxruntime
import time
import numpy as np
import cv2
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from inference_utils.model_exchange import ModelExchange
# logger to capture errors, warnings, and other information during the build and inference phases
# TRT_LOGGER = trt.Logger()

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
                 checkpoint_path_detect: str) -> None:
        self.checkpoint_path_classify = checkpoint_path_classify
        self.checkpoint_path_detect = checkpoint_path_detect
        self.model_classify = torch.load(checkpoint_path_classify, map_location=torch.device("cpu")) 
        self.model_detect = torch.load(checkpoint_path_detect, map_location=torch.device("cpu")) 
        self.onnxruntime = None
        self.onnxruntime_detect= None
        self.frames = []
        self.val_dataloader = None
        self.spatial_transform = None
        self.temporal_transform_test = None
        self.model_exchange_detect = None
        self.model_exchange_classify = None
        self.need_classify_video = []
        self._get_spatial_temporal()
        # self._build_onnxruntime()
        self._is_gpu_available()
        self.cap = cv2.VideoCapture(0)
        self._get_device_info()
        self._init_model_exchange_detect()
        self._init_model_exchange_classify()

    def _init_model_exchange_detect(self):
        weights_name = self.checkpoint_path_detect
        input_shape = (1, 3, 224, 224)
        self.model_exchange_detect = ModelExchange(weights_name, input_shape, 'fp32')

    def _init_model_exchange_classify(self):
        weights_name = self.checkpoint_path_classify
        input_shape = (1, 8, 3, 224, 224)
        self.model_exchange_classify = ModelExchange(weights_name, input_shape, 'fp32')


    def _get_device_info(self):
        print("摄像头帧率: ", self.cap.get(cv2.CAP_PROP_FPS))


    def capture_run(self):
        if not self.cap.isOpened():
            raise Exception("Could not open camera.")
        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('frame', frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_image = Image.fromarray(frame_rgb)
                rgb_cache = rgb_image.convert("RGB")
                detect_result = self.inference_detect_pth(rgb_cache)
                if detect_result == 1:
                    self.need_classify_video.append(rgb_cache)
                if detect_result == 0 and len(self.need_classify_video) > 8:
                    print("----------------------------- classify -----------------")
                    self.inference_pth()
                    self.need_classify_video = []
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def _is_gpu_available(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_detect = self.model_detect.to(device)
        self.model_classify = self.model_classify.to(device)
        if device.type == 'cuda':
            self.model_detect.cuda()
            self.model_classify.cuda()
        else:
            self.model_detect = self.model_detect.module
            self.model_classify = self.model_classify.module
        self.model_detect.eval()
        self.model_classify.eval()


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
    

    def _load_dataloader(self):
        inference_data = need_to_classify_video(self.need_classify_video, spatial_transform=self.spatial_transform,
                                                        temporal_transform=self.temporal_transform_test)
        return inference_data.get_inference_data()
    

    def inference_pth(self):
        # start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model_classify(self._load_dataloader())
            # outputs = self.model_classify(self.val_dataloader)
            outputs = outputs.view(1, 1, -1)
            outputs = F.softmax(outputs, 2)
            # 找到概率最大值的索引，这就是预测值
            _, predicted_class = torch.max(outputs, 2)
            print(f'\r预测的类别是: {predicted_class[0][0]}', end='')
        # end_time = time.perf_counter()
        # execution_time = end_time - start_time
        # print(f"pth 代码运行时间: {execution_time}秒")
        return int(predicted_class[0][0])
    
    
    def inference_detect_pth(self, image):
        # start_time = time.perf_counter()
        with torch.no_grad():
            image = self._preprocess_image(image)
            outputs = self.model_detect(image)
            outputs = outputs.view(1, -1)
            outputs = F.softmax(outputs, 1)
            # 找到概率最大值的索引，这就是预测值
            _, predicted_class = torch.max(outputs, 1)
            # print(f'\r预测的类别是: {predicted_class[0]}', end='')
        # end_time = time.perf_counter()
        # execution_time = end_time - start_time
        # 计算代码运行时间
        # print(f"pth 代码运行时间: {execution_time}秒")
        return int(predicted_class[0])
    
    def _inference_detect_trt(self, image):
        image = self._preprocess_image(image)
        trt_result = self.inference_detect_with_trt(image)
        return trt_result
    
    def inference_detect_with_trt(self, img_in):
        h_input = cuda.pagelocked_empty(trt.volume(self.context.get_binding_shape(0)), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(self.context.get_binding_shape(1)), dtype=np.float32)
        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        # set the host input data
        # h_input = img_in
        np.copyto(h_input, img_in.ravel())
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        self.context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # outputs = h_output.view(1, -1)
        # outputs = F.softmax(outputs, 1)
        # 找到概率最大值的索引，这就是预测值
        _, predicted_class = torch.max(torch.from_numpy(np.array([h_output])), 1)
        print(f'\r预测的类别是: {predicted_class[0]}', end='')
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
        print(f'\ronnx 预测的类别是: {predicted_class[0][0]}', end='')
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"\ronnx 代码运行时间: {execution_time}秒", end='')
        return int(predicted_class[0][0])



if __name__ == "__main__":
    classify_weights_name = "mobilenetv2classify.pth"
    detect_weights_name = "mobilenetv3_smalldetect.pth"
