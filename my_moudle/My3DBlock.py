
from others.params import *
from my_moudle.local_attention import *
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D


class My3DBlock(nn.Module):
    def __init__(self, channel):
        super(My3DBlock, self).__init__()
        self.inter_channel = channel
        # self.conv_q = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=(1, 1, 1), bias=False, padding=(1, 0, 0))
        # self.conv_k = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=(1, 1, 1), bias=False, padding=(1, 0, 0))
        # self.conv_v = nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=(1, 1, 1), bias=False, padding=(1, 0, 0))
        self.conv_q = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
        self.conv_k = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
        self.conv_v = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.conv_before = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
        self.conv_after = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
        self.positionalEncoding3D = PositionalEncoding3D(1)  # 3D ：  [Batch size, Depth, Height, Width, Channels]\
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // args.clip_len
        x_3d = x.reshape(n_batch, args.clip_len, c, h, w).contiguous()
        x_3d = x_3d.permute(0, 1, 3, 4, 2).contiguous()  # n, t, h, w, c
        x_3d = x_3d.mean(4, keepdim=True).contiguous()  # n, t, h, w, 1

        x_3d = self.positionalEncoding3D(x_3d)
        x_3d = x_3d.to(x.device)

        x_3d = x_3d.permute(0, 4, 1, 2, 3).contiguous()  # n ,1 ,t ,h, w

        x_p3 = x_3d.permute(0, 2, 1, 3, 4)
        x3_plus0, _ = x_p3.split([args.clip_len - 1, 1], dim=1)
        _, x3_plus1 = x_p3.split([1, args.clip_len - 1], dim=1)
        x_p3 = x3_plus1 - x3_plus0                       # n,t,1,h,w
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3 = x_p3.permute(0, 2, 1, 3, 4).contiguous()

        # print(self.conv_q.weight.device)
        x_q = self.conv_q(x_p3)
        x_k = self.conv_k(x_p3)
        x_v = self.conv_v(x_3d)

        attention = (x_q @ x_k.permute(0, 1, 2, 4, 3)) * (h ** -1)  # n, 1, t, h, w
        # attention = attention.softmax(dim=1)

        x_out = (attention @ x_v)  # n, 1, t, h, w

        x_out = self.conv_after(x_out)

        x_out = x_out.permute(0, 2, 1, 3, 4).contiguous()
        x_out = x_out.reshape(nt, 1, h, w).contiguous()  # nt, 1, h, w
        # x_out = self.sigmoid(x_out)
        # x_out = self.relu(x_out)


        return x_out  # nt, 1, h, w




# class My3DAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(My3DAttention, self).__init__()
#         self.in_channels = in_channels
#         self.conv_q = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
#         self.conv_k = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
#         self.conv_v = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
#         self.positionalEncoding3D = PositionalEncoding3D(1)  # 3D ：  [Batch size, Depth, Height, Width, Channels]\
#         self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
#         self.softmax = nn.Softmax(dim=-1)
#         # self.sigmoid = nn.sigmoid()
#
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // args.clip_len
#         x_3d = x.reshape(n_batch, args.clip_len, c, h, w).contiguous()
#         x_3d = x_3d.permute(0, 1, 3, 4, 2).contiguous()  # n, t, h, w, c
#         x_3d = x_3d.mean(4, keepdim=True).contiguous()  # n, t, h, w, 1
#
#         x_3d = self.positionalEncoding3D(x_3d)
#         x_3d = x_3d.to(x.device)
#         x_3d = x_3d.permute(0, 4, 1, 2, 3).contiguous()  # n ,1 ,t ,h, w
#
#         x_p3 = x_3d.permute(0, 2, 1, 3, 4)
#         x3_plus0, _ = x_p3.split([args.clip_len - 1, 1], dim=1)
#         _, x3_plus1 = x_p3.split([1, args.clip_len - 1], dim=1)
#         x_p3 = x3_plus1 - x3_plus0                       # n,t,1,h,w
#         x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
#         x_p3 = x_p3.permute(0, 2, 1, 3, 4).contiguous()
#
#
#         x_q = self.conv_q(x_p3)   # n, 1, t, h ,w
#         x_k = self.conv_k(x_p3)
#         x_v = self.conv_v(x_3d)
#
#         x_q = x_q.view(n_batch, -1, args.clip_len * h * w).permute(0, 2, 1)
#         x_k = x_k.view(n_batch, -1, args.clip_len * h * w)
#
#         energy = torch.bmm(x_q, x_k)
#         attention = self.softmax(energy)
#         x_v = x_v.view(n_batch, -1, args.clip_len * h * w)
#         out = torch.bmm(x_v, attention.permute(0, 2, 1))
#         out = out.view(n_batch, 1, args.clip_len, h, w).permute(0, 2, 1, 3, 4)
#         out = out.reshape(nt, 1, h, w)
#         # out = self.sigmoid(out)
#
#         return out




# if __name__=='__main__':
#     model = My3DBlock(64)
#
#     input = torch.randn(16, 64, 56, 56)
#     out = model(input)
#     print(out.shape)
