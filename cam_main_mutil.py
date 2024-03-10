# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch.nn.utils import clip_grad_norm_
from CAM3D.ops.dataset import TSNDataSet
from CAM3D.ops.models import TSN
from CAM3D.ops.transforms import *
from CAM3D.opts import parser
from CAM3D.ops.utils import AverageMeter, accuracy
from CAM3D.ops.temporal_shift import make_temporal_pool
from tensorboardX import SummaryWriter
from CAM3D.ops.CAM3D_module import ClassAttentionMapping
import cv2
from others.train import *
from models.models import *
from others.params import *
from deals.load_dataset import * 
from PIL import Image



def get_path():
    # 获取当前文件路径
    current_file = os.path.abspath(__file__)
    # 获取所在目录路径
    current_directory = os.path.dirname(current_file)
    return current_directory



def merge_images_in_folders(folder_paths, output_path):
    images_per_folder = 8
    gap_size = 10  # 调整图片之间的间隙大小
    row_images = []

    for folder_path in folder_paths:
        row_images_folder = []
        for i in range(images_per_folder):
            image_path = f"{folder_path}/{i+1}.jpg"  # 假设图片名称为image_1.jpg, image_2.jpg, ...
            image = Image.open(image_path)
            row_images_folder.append(image)
        row_images.append(row_images_folder)

    max_widths = [max(image.width for image in row) for row in row_images]
    total_width = sum(max_widths) + (images_per_folder - 1) * gap_size
    max_height = max(image.height for row in row_images for image in row)

    new_image = Image.new("RGB", (total_width, max_height * len(row_images)))

    y_offset = 0
    for row_images_folder, max_width in zip(row_images, max_widths):
        x_offset = 0
        for image in row_images_folder:
            new_image.paste(image, (x_offset, y_offset))
            x_offset += max_width + gap_size
        y_offset += max_height

    new_image.save(output_path)



def cam_process(train_loader, norm_param, action_net_model, mrc_model, iff_model, mrc_and_iff_model):
    # 使用示例 # 五个文件夹的路径列表
    folder_paths = ["cam_results/srcimages",
                    "cam_results/action_net_camimages", 
                    "cam_results/mrc_camimages",
                    "cam_results/iff_camimages", 
                    "cam_results/mrc_and_iff_camimages"]  
    # classifier   layer4
    action_net_cam = ClassAttentionMapping(action_net_model, 'base_model.classifier', 'new_fc.weight', norm_param)
    mrc_cam = ClassAttentionMapping(mrc_model, 'base_model.classifier', 'new_fc.weight', norm_param)
    iff_cam = ClassAttentionMapping(iff_model, 'base_model.classifier', 'new_fc.weight', norm_param)
    mrc_and_iff_cam = ClassAttentionMapping(mrc_and_iff_model, 'base_model.classifier', 'new_fc.weight', norm_param)

    record_dict = {key: 0 for key in range(0, 83)}
    total = len(record_dict)

    for input, _, target in train_loader:
        for label in target:
            label = int(label)
            if record_dict[label] == 0:
                total -= 1
                record_dict[label] = 1
                action_net_cam_out = action_net_cam(input[0].cuda())
                mrc_cam_out = mrc_cam(input[0].cuda())
                iff_cam_out = iff_cam(input[0].cuda())
                mrc_and_iff_cam_out = mrc_and_iff_cam(input[0].cuda())

                for i, (src_img, dst_img) in enumerate(zip(*action_net_cam_out)):
                    cv2.imwrite("cam_results/srcimages/" + str(i + 1) + ".jpg", src_img)
                    cv2.imwrite("cam_results/action_net_camimages/" + str(i + 1) + ".jpg", dst_img)
                for i, (src_img, dst_img) in enumerate(zip(*mrc_cam_out)):
                    cv2.imwrite("cam_results/mrc_camimages/" + str(i + 1) + ".jpg", dst_img)
                for i, (src_img, dst_img) in enumerate(zip(*iff_cam_out)):
                    cv2.imwrite("cam_results/iff_camimages/" + str(i + 1) + ".jpg", dst_img)
                for i, (src_img, dst_img) in enumerate(zip(*mrc_and_iff_cam_out)):
                    cv2.imwrite("cam_results/mrc_and_iff_camimages/" + str(i + 1) + ".jpg", dst_img)
                
                # 合并后的图片保存路径
                output_path = "cam_results/mergre_results/" + str(label) + ".jpg" 
                merge_images_in_folders(folder_paths, output_path)
            else:
                continue
            if total == 0:
                break

 





def main():
    # 加载模型                                       
    print("load model")

    dir = get_path()
    
    checkpoint_path_action_net = dir + "/weights/mobilenetv2classify.pth"
    checkpoint_path_mrc = dir + "/weights/mobilenetv2classify.pth"
    checkpoint_path_iff = dir + "/weights/mobilenetv2classify.pth"
    checkpoint_path_mrc_and_iff = dir + "/weights/mobilenetv2classify.pth"

    action_net_model = torch.load(checkpoint_path_action_net, map_location=torch.device("cpu")).module 
    mrc_model = torch.load(checkpoint_path_mrc, map_location=torch.device("cpu")).module 
    iff_model = torch.load(checkpoint_path_iff, map_location=torch.device("cpu")).module  
    mrc_and_iff_model = torch.load(checkpoint_path_mrc_and_iff, map_location=torch.device("cpu")).module  

    action_net_model = action_net_model.cuda()    
    mrc_model = mrc_model.cuda()
    iff_model = iff_model.cuda()
    mrc_and_iff_model = mrc_and_iff_model.cuda()


    input_mean = action_net_model.input_mean
    input_std = action_net_model.input_std
    norm_param = (input_mean, input_std)
    
    # 加载数据集
    train_loader, val_loader = load_dataset()

    cam_process(val_loader, norm_param, action_net_model, mrc_model, iff_model, mrc_and_iff_model)

    

if __name__ == '__main__':
    main()
