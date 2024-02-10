import os
import torch
from typing import Any
import onnx
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F
import time
from inference_utils.load_inference_data import *
from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 
from onnxsim import simplify


class ModelExchange:
    def __init__(self, weights_name: str, input_shape: tuple, precision: str) -> None:
        self.weights_path = weights_name
        self.input_shape = input_shape
        self.precision = precision
        self.current_directory = self._init_get_current_path()
        self._exchange_run()
        self.get_content()

    def _init_get_current_path(self):
        current_file = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file)
        print("current_directory : ", os.path.dirname(current_directory))
        return os.path.dirname(current_directory)

    def _check_onnx_is_succesful(self):
        onnx_model = onnx.load(self.onnx_save_path)
        onnx.checker.check_model(onnx_model)
    
    def _convert_to_onnx(self):
        self.onnx_save_path = self.weights_path.split('.')[0] + self.precision + '.onnx'
        model_onnx = torch.load(self.weights_path, map_location=torch.device("cpu"))
        model_onnx.eval()
        dummpy_data = torch.randn(self.input_shape)
        torch.onnx.export(
            model_onnx,
            dummpy_data,
            self.onnx_save_path,
            opset_version = 12,
            input_names = ['input'],
            output_names = ['output']
        )
        self._check_onnx_is_succesful()
        print("onnx has been saved to : ", self.onnx_save_path)
        self._convert_onnx_to_simple()

    def _convert_onnx_to_simple(self):
        onnx_model = onnx.load(self.onnx_save_path)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, self.onnx_save_path)

    def _convert_to_trt(self):
        TRT_LOGGER = trt.Logger()
        self.engine_file_path = self.onnx_save_path.split('.onnx')[0] + '.trt'
        EXPLICIT_BATCH = [1]
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network,
                                                                                                            TRT_LOGGER) as parser:
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 20
            builder.max_batch_size = 1
            # print(network)
            with open(self.onnx_save_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print('parser.get_error(error)', parser.get_error(error))
            last_layer = network.get_layer(network.num_layers - 1)
            network.mark_output(last_layer.get_output(0))
            network.get_input(0).shape = list(self.input_shape)
            print('Completed parsing of ONNX file')
            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            config.add_optimization_profile(profile)
            engine  = builder.build_engine(network, config)
            with open(self.engine_file_path, "wb") as f:
                f.write(engine.serialize())
                print('save  trt success!! : ', self.engine_file_path)

    def get_content(self):
        self._load_engine()
        self.context = self.engine.create_execution_context()

    def _load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_file_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
    
    def _load_dataloader(self, need_classify_video):
        inference_data = need_to_classify_video(need_classify_video, spatial_transform=self.spatial_transform,
                                                        temporal_transform=self.temporal_transform_test)
        return inference_data.get_inference_data()
    
    def _exchange_run(self):
        self._convert_to_onnx()
        self._convert_to_trt()



# if __name__ == "__main__":
#     weights_name = "mobilenetv3_smalldetect.pth"
#     input_shape = (1, 3, 224, 224)
#     model_exchange = ModelExchange(weights_name, input_shape, 'fp32')

