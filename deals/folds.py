from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

# unfolding模块实质上就是将[B, C, H, W]-->[BP, N, C]
# 就是将原来每个pixel与其他所有pixel做MSA变成了patch_h*patch_w份，每份内部做MSA，分批次输入进Transformer中。
def unfolding(x: Tensor) -> Tuple[Tensor, Dict]:
    patch_w, patch_h = 2, 2
    patch_area = patch_w * patch_h  # 4
    batch_size, in_channels, orig_h, orig_w = x.shape
    # 向上取整，若是不能整除，则将feature map尺寸扩大到能整除
    new_h = int(math.ceil(orig_h / patch_h) * patch_h)
    new_w = int(math.ceil(orig_w / patch_w) * patch_w)

    interpolate = False
    if new_w != orig_w or new_h != orig_h:
        # Note: Padding can be done, but then it needs to be handled in attention function.
        # 若是扩大feature map尺寸，则用双线性插值法
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        interpolate = True

    # number of patches along width and height
    # patches的数量
    num_patch_w = new_w // patch_w  # n_w
    num_patch_h = new_h // patch_h  # n_h
    num_patches = num_patch_h * num_patch_w  # N

    # [B, C, H, W] -> [B * C * num_patch_h, patch_h, num_patch_w, patch_w]
    x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
    # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
    x = x.transpose(1, 2)
    # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    # P为patches面积大小，N为patches数量
    x = x.reshape(batch_size, in_channels, num_patches, patch_area)
    # [B, C, N, P] -> [B, P, N, C]
    x = x.transpose(1, 3)
    # [B, P, N, C] -> [BP, N, C]
    x = x.reshape(batch_size * patch_area, num_patches, -1)

    info_dict = {
        "orig_size": (orig_h, orig_w),
        "batch_size": batch_size,
        "interpolate": interpolate,
        "total_patches": num_patches,
        "num_patches_w": num_patch_w,
        "num_patches_h": num_patch_h,
    }

    return x, info_dict


# unfolding模块实质上就是将[BP, N, C]-->[B, C, H, W]
def folding(x: Tensor, info_dict: Dict) -> Tensor:
    n_dim = x.dim()
    assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
        x.shape
    )
    # [BP, N, C] --> [B, P, N, C]
    x = x.contiguous().view(
        info_dict["batch_size"], 4, info_dict["total_patches"], -1
    )

    batch_size, pixels, num_patches, channels = x.size()
    num_patch_h = info_dict["num_patches_h"]
    num_patch_w = info_dict["num_patches_w"]

    # [B, P, N, C] -> [B, C, N, P]
    x = x.transpose(1, 3)
    # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
    x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, 3, 3)
    # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
    x = x.transpose(1, 2)
    # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
    x = x.reshape(batch_size, channels, num_patch_h * 3, num_patch_w * 3)
    if info_dict["interpolate"]:
        x = F.interpolate(
            x,
            size=info_dict["orig_size"],
            mode="bilinear",
            align_corners=False,
        )
    return x
