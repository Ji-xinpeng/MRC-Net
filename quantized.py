from torchvision.models import mobilenet_v3_small as mobilenet_v3_small
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import get_default_qconfig
import os
import time
from deals.load_dataset import *
from others.train import *
from torchsummary import summary
from models.models import *
import h5py
from detect_utils.detect_utils import *
import torch.quantization
from tqdm import tqdm



def load_classift_model(checkpoint_path, device):
    cudnn.benchmark = True
    model = TSN(params['num_classes'], args.clip_len, 'RGB', 
                        is_shift = args.is_shift,
                        base_model="mobilenetv2", 
                        shift_div = args.shift_div, 
                        img_feature_dim = 224,
                        consensus_type='avg',
                        fc_lr5 = True)

    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    print("load checkpoint {}".format(checkpoint_path))
    # print(pretrained_dict.keys())
    model.load_state_dict(pretrained_dict['state_dict'])
    # model = nn.DataParallel(model)  # multi-Gpu
    model = model.to(device)
    return model


def load_detect_model(checkpoint_path, device):
    cudnn.benchmark = True
    model = mobilenet_v3_small(pretrained=False)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    print("load checkpoint {}".format(checkpoint_path))
    model.load_state_dict(pretrained_dict['state_dict'])
    # model = nn.DataParallel(model)  # multi-Gpu
    model = model.to(device)
    return model

# def quantization(model, checkpoint_path, train_loader, int8orfp16, device):
#     print("-----  weights path ----", checkpoint_path)
#     model.eval()
#     # 指定量化配置, - "fbgemm" 对应 x86 架构，"qnnpack" 对应移动架构
#     model.qconfig = torch.quantization.get_default_qconfig('x86')
#     # 准备模型进行后训练静态量化
#     model = torch.quantization.prepare(model)
#     # 运行模型，执行校准
#     for step, inputs in tqdm(enumerate(train_loader), total=len(train_loader), desc="quantization"):
#         rgb, depth, labels = inputs[0], inputs[1], inputs[2]
#         rgb = rgb.to(torch.device('cuda'))
#         model(rgb)
#     # 将量化方案应用到模型上
#     model_quantized = torch.quantization.convert(model)
#     # 保存量化模型
#     save_path = checkpoint_path.split('.')[0] + "quantized_model" + int8orfp16 + ".pth.tar"
#     print("----- quantization weights saved path ----", save_path)
#     checkpoint = {
#         'state_dict': model_quantized.state_dict(),
#         }
#     torch.save(checkpoint, save_path)
#     return save_path


def quantization(model, checkpoint_path, train_loader, int8orfp16, device):
    from torch.ao.quantization import ( 
        get_default_qconfig_mapping, 
        get_default_qat_qconfig_mapping, 
        QConfigMapping, 
    ) 
    import torch.quantization.quantize_fx as quantize_fx 
    import copy 
    model_to_quantize = copy.deepcopy(model).to(device) 
    model_to_quantize.eval() 
    qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig) 
    example_inputs = next(iter(train_loader))[0] 
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping,  
                    example_inputs) 
    model_quantized_dynamic = quantize_fx.convert_fx(model_prepared)
    save_path = checkpoint_path.split('.')[0] + "quantized_model" + int8orfp16 + ".pth"
    checkpoint = {
        'state_dict': model_quantized_dynamic.state_dict(),
        }
    torch.save(checkpoint, save_path)
    return save_path


def parse_select_weights_path(name: str):
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    checkpoint_path = current_directory + '/weights/' + name
    return checkpoint_path


def evaluate(model, data_loader, device):
    correct = 0
    total = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(data_loader), total=len(data_loader), desc="evaluate"):
            rgb, depth, labels = inputs[0], inputs[1], inputs[2]
            rgb = rgb.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(rgb)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"推理代码运行时间: {execution_time}秒,  一共  {total} 个数据， 单个数据 {execution_time / total}秒")
    acc = 100 * correct / total
    return acc, execution_time / total



def quantization_run(name, train_loader, test_loader, model, device, path, int8orfp16):
    quantization_before_acc, infer_time_before = evaluate(model, test_loader, device)
    print(f"量化之前准确率：{quantization_before_acc}, 推理时间为 {infer_time_before}")

    quantization_path = quantization(model, path, train_loader, int8orfp16, torch.device('cpu'))
    print("quantization_path : ", quantization_path)
    if "detect" in quantization_path:
        quantization_model = load_detect_model(quantization_path, device)
    elif "classify" in quantization_path:
        quantization_model = load_classift_model(quantization_path, device)

    quantization_after_acc, infer_time_after = evaluate(quantization_model, test_loader, device)
    print(f"量化之前准确率：{quantization_after_acc}, 推理时间为 {infer_time_after}")
    print(f' {name} Accuracy 下降了 ====== : {quantization_before_acc - quantization_after_acc}')



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # "fp16"  "int8" 
    int8orfp16 = "int8"   

    print("----------------------------  classify quantization  -----------------------------")
    classify_name = "mobilenetv2classifyonlystate.pth.tar"
    classify_path = parse_select_weights_path(classify_name)
    train_loader, test_loader = load_dataset()
    model_classify = load_classift_model(classify_path, device)
    quantization_run(classify_name, test_loader, test_loader, model_classify, device, classify_path, int8orfp16)

    print("----------------------------  detect quantization  -----------------------------")
    # detect_name = "mobilenetv3_samlldetectonlystate.pth.tar"
    # detect_path = parse_select_weights_path(detect_name)
    # train_dataloader, val_dataloader = get_detect_dataloader()
    # model_detect = load_detect_model(detect_path, device)
    # quantization_run(detect_name, train_dataloader, val_dataloader, model_detect, device, detect_path, int8orfp16)

    