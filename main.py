import torch.nn
from torchsummary import summary
from detect_utils.detect_utils import *
from others.params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
device = 'cuda'

def main():
    # 记录准确率
    record = Record()

    # 加载数据集
    if args.is_detector_classify == "classify":
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
                            pretrain='imagenet',  # 'imagenet' or False
                            consensus_type='avg',
                            fc_lr5=True)
        # 预训练
        if params['pretrained'] is not None:
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
        policies = None
        input_data_shape = (8, 3, 224, 224)
    else:
        train_dataloader, val_dataloader = get_detect_dataloader()
        print("--------------------------------- detect train -------------------------------")

        if args.is_mobile_v3_small:
            from torchvision.models import mobilenet_v3_small as mobilenet_v3_small
            model = mobilenet_v3_small(pretrained=True)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, 2)
        else:
            from torchvision.models import mobilenet_v2 as mobilenet_v2
            model = mobilenet_v2(pretrained=True)
            # 修改分类器以适应自己的数据集
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)

        policies = get_detect_model_policies(model)
        input_data_shape = (3, 224, 224)


    # 计算模型参数数量
    print("模型的参数量:", sum(p.numel() for p in model.parameters()))
    summary(model, input_data_shape, device="cpu")


    my_train(model, train_dataloader, val_dataloader, device, record, policies)

    # 画图
    record.get_size()
    plot(record)


if __name__ == '__main__':
    main()