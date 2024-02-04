import argparse
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0')

    # args for dataloader
    parser.add_argument('--is_train', action="store_true", default=True)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=36)
    parser.add_argument('--clip_len', type=int, default=8)

    # args for preprocessing
    parser.add_argument('--initial_scale', type=float, default=1,
                        help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int,
                        help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float,
                        help='Scale step for multiscale cropping')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', type=float, default=[15, 30, 45], nargs="+",
                        help='lr steps for decreasing learning rate')
    parser.add_argument('--clip_gradient', '--gd', type=int, default=20, help='gradient clip')
    parser.add_argument('--shift_div', default=8, type=int)
    parser.add_argument('--is_shift', action="store_true", default=True)
    parser.add_argument('--npb', action="store_true")
    parser.add_argument('--pretrain', type=str, default='imagenet')  # 'imagenet' or False
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--base_model', default='mobilenetv2', type=str)  # resnet50   mobilenetv2    BNInception
    parser.add_argument('--dataset', default='EgoGesture', type=str)   # sthv2    EgoGesture    jester
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', default='imagenet', type=str)
    parser.add_argument('--use_video_swin_transformer', default=False, type=str)
    parser.add_argument('--change_resnet_head', default=False, type=str)  # 改为 True 就改变 resnet 开头几层网络
    parser.add_argument('--num_heads', default=1, type=str) # 自注意力机制头数
    parser.add_argument('--reduce_num', default=32, type=str)  # my_block 通道减少的数量
    parser.add_argument('--if_need_load_dataset', default=False, type=str)  #
    parser.add_argument('--begin_split', default=1, type=str)  # my_block
    parser.add_argument('--if_get_data_and_label', default=True, type=str)
    parser.add_argument('--is_detector_classify', default="detect", type=str) # classify detect

    args = parser.parse_args()
    return args


args = parse_opts()


# 获取当前文件路径
current_file = os.path.abspath(__file__)
# 获取所在目录路径
current_directory = os.path.dirname(current_file)
# 获取上一级目录路径
parent_directory = os.path.dirname(current_directory)


params = dict()
if args.dataset == 'EgoGesture' and args.base_model == "resnet50":
    params['num_classes'] = 83
    params['pretrained'] = parent_directory + '/weights/clip_len_8frame_sample_rate_1_checkpoint.pth.tar'
elif args.dataset == 'jester' and args.base_model == "resnet50":
    params['num_classes'] = 27
    params['pretrained'] = parent_directory + '/weights/clip_len_8frame_sample_rate_1_checkpoint.pth (copy).tar'
elif args.dataset == 'sthv2' and args.base_model == "resnet50":
    params['num_classes'] = 174

if args.dataset == 'EgoGesture' and args.base_model == "mobilenetv2":
    params['num_classes'] = 83
    params['pretrained'] = None
elif args.dataset == 'jester' and args.base_model == "mobilenetv2":
    params['num_classes'] = 27
    params['pretrained'] = None
elif args.dataset == 'sthv2':
    params['num_classes'] = 174

params['epoch_num'] = args.epochs
params['batch_size'] = args.batch_size
params['num_workers'] = args.num_workers
params['learning_rate'] = args.lr

if args.is_detector_classify == "detect":
    params['batch_size'] = args.batch_size * 8
    args.lr_steps = [1, 2]
    params['epoch_num'] = 3

params['momentum'] = 0.9
params['weight_decay'] = args.weight_decay
params['display'] = 100
# params['pretrained'] = None
# params['pretrained'] = './clip_len_8frame_sample_rate_1_checkpoint.pth.tar'
params['recover_from_checkpoint'] = None
params['log'] = 'log-{}'.format(args.dataset)
params['save_path'] = '{}-{}'.format(args.dataset, args.base_model)
params['clip_len'] = args.clip_len
params['frame_sample_rate'] = 1
