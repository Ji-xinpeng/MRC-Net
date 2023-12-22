import torch.nn

from deals.load_dataset import *
from others.train import *
from torchsummary import summary
from models.models import *
import h5py

# 加载数据集
train_dataloader, val_dataloader = load_dataset()


# 加载模型
print("load model")
model = TSN(params['num_classes'], args.clip_len, 'RGB',
                      is_shift=args.is_shift,
                      partial_bn=args.npb,
                      base_model=args.base_model,
                      shift_div=args.shift_div,
                      dropout=args.dropout,
                      img_feature_dim=224,
                      pretrain='False',  # 'imagenet' or False
                      consensus_type='avg',
                      fc_lr5=True)


# 开始训练
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
device = 'cuda'

# 预训练
if params['pretrained'] is not None:
    print("----------------------------------------------------")
    print(params['pretrained'])
    checkpoint = torch.load(params['pretrained'], map_location='cpu')
    try:
        model_dict = model.module.state_dict()
    except AttributeError:
        model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and 'fc' not in k}
    print("load pretrained model {}".format(params['pretrained']))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# 计算模型参数数量
print("模型的参数量:", sum(p.numel() for p in model.parameters()))
summary(model, (8, 3, 224, 224), device="cpu")

# 记录准确率
record = Record()

my_train(model, train_dataloader, val_dataloader, device, record)

# 画图
record.get_size()
plot(record)
