from others.train import *
from models.models import *
from deals.load_dataset import *
from torchsummary import summary
import torch
from thop import profile
from PIL import Image
import torchvision
import torchvision.transforms as transforms


# 加载数据集
train_dataloader, val_dataloader = load_dataset()

# 加载数据集
x = torch.randn(1, 8, 3, 224, 224)

# 加载模型
print("load model")
model = TSN(params['num_classes'], args.clip_len, 'RGB',
                      is_shift=args.is_shift,
                      partial_bn=args.npb,
                      base_model=args.base_model,
                      shift_div=args.shift_div,
                      dropout=args.dropout,
                      img_feature_dim=224,
                      pretrain='imagenet',  # 'imagenet' or False
                      consensus_type='avg',
                      fc_lr5=True)


# 计算模型参数数量
print("模型的参数量:", sum(p.numel() for p in model.parameters()))
summary(model, (args.clip_len, 3, 224, 224), -1, device="cpu")

y = model(x)
print("经过模型输出的张量维度是", y.shape)

print("可用的GPU数量：", torch.cuda.device_count())

x = torch.randn(1, args.clip_len, 3, 224, 224)
flops, params = profile(model, (x,))
print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

