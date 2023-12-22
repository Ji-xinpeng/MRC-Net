import torch
import numpy as np
import torch.nn.functional as F


def join_pictures(x):
    """
    把 9 张图片 拼接成 1 张
    :param x: 是五维的矩阵，如 (4, 9, 3, 28, 28)
    :return: 拼接完成的大矩阵
    """
    # print(x.shape)
    matrix1 = x[:, 0, :, :, :]
    matrix2 = x[:, 1, :, :, :]
    matrix3 = x[:, 2, :, :, :]
    matrix4 = x[:, 3, :, :, :]
    matrix5 = x[:, 4, :, :, :]
    matrix6 = x[:, 5, :, :, :]
    matrix7 = x[:, 6, :, :, :]
    matrix8 = x[:, 7, :, :, :]
    matrix9 = x[:, 8, :, :, :]

    matrix_top = torch.cat((matrix1, matrix2, matrix3), dim=2)
    matrix_middle = torch.cat((matrix4, matrix5, matrix6), axis=2)
    matrix_bottom = torch.cat((matrix7, matrix8, matrix9), axis=2)
    matrix_combined = torch.cat((matrix_top, matrix_middle, matrix_bottom), axis=3)

    return matrix_combined



def split_pictures(x):
    """
    把 1 张图片 分成 9 张
    :param x: 是五维的矩阵，如 (4*8, 3, 28, 28)
    :return: 拼接完成的大矩阵
    """
    nt, c, h, w = x.shape
    h = h // 2
    w = w // 2
    f = c // 4
    # matrix1 = x[:, :, 0:h, 0:w]
    # matrix2 = x[:, :, 0:h, w:]
    # matrix3 = x[:, :, h:, 0:w]
    # matrix4 = x[:, :, h:, w:]
    matrix1 = x[:, :f, :, :]
    matrix2 = x[:, f:2*f, :, :]
    matrix3 = x[:, 2*f:3*f, :, :]
    matrix4 = x[:, 3*f:, :, :]
    # print(matrix1.shape)
    # print(matrix2.shape)
    # print(matrix3.shape)
    # print(matrix4.shape)

    return matrix1, matrix2, matrix3, matrix4




def patchmerging(x):
    """
    x: nt, h, w, c
    """
    # padding
    # 如果输入feature map的H，W不是2的整数倍，需要进行padding
    nt, h, w, c = x.shape

    pad_input = (h % 2 == 1) or (w % 2 == 1)
    if pad_input:
        # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

    x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
    x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
    x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
    x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
    x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
    x = x.view(nt, -1, 4 * c)  # [B, H/2*W/2, 4*C]

    return x, pad_input


def restore_pictures(x, pad_input):
    nt, h, w, c = x.shape
    # 是否存在填充
    # pad_input = (h % 2 == 1) or (w % 2 == 1)
    # 对x进行还原
    x0 = x[:, :, :, 0:c // 4]
    x1 = x[:, :, :, c // 4:c // 2]
    x2 = x[:, :, :, c // 2:3 * c // 4]
    x3 = x[:, :, :, 3 * c // 4:c]

    output = torch.zeros((nt, h * 2, w * 2, c // 4), dtype=x.dtype, device=x.device)

    output[:, 0::2, 0::2, :] = x0
    output[:, 1::2, 0::2, :] = x1
    output[:, 0::2, 1::2, :] = x2
    output[:, 1::2, 1::2, :] = x3

    if pad_input:
        # 对输出进行裁剪以去除填充部分
        output = output[:, :h * 2 -1, :w * 2 - 1, :]

    return output

