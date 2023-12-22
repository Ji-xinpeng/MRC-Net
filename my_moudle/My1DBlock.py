
from others.params import *
from my_moudle.local_attention import *
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D


# class My1DBlock(nn.Module):
#     def __init__(self, channel):
#         super(My1DBlock, self).__init__()
#         self.inter_channel = channel
#         self.conv_q = nn.Conv1d(self.inter_channel, self.inter_channel // args.reduce_num, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
#         self.conv_k = nn.Conv1d(self.inter_channel, self.inter_channel // args.reduce_num, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
#         self.conv_v = nn.Conv1d(self.inter_channel, self.inter_channel // args.reduce_num, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
#         self.conv_before = nn.Conv1d(self.inter_channel, self.inter_channel // args.reduce_num, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
#         self.conv_after = nn.Conv1d(self.inter_channel // args.reduce_num, self.inter_channel, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.positionalEncoding1D = PositionalEncoding1D(self.inter_channel)   # 1D ： [Batch size, Sequence length, Channels]
#
#
#     def forward(self, x):
#         x_1d = self.avg_pool(x)   # nt, c, 1, 1
#         nt, c, h, w = x_1d.size()
#         n_batch = nt // args.clip_len
#
#         x_1d = x_1d.reshape(nt, c).reshape(n_batch, args.clip_len, c).contiguous()  # n, t, c
#         # 1D ： [Batch size, Sequence length, Channels]
#         x_1d = self.positionalEncoding1D(x_1d)
#         x_1d = x_1d.to(x.device)
#
#         x_change = torch.roll(x_1d, shifts=args.begin_split, dims=1)
#         x_change = x_change.permute(0, 2, 1)
#         x_change = self.conv_before(x_change)   # n, t, c // 16
#
#         x_1d = x_1d.permute(0, 2, 1)   # n,c t
#         x_q = self.conv_q(x_1d)
#         x_k = self.conv_k(x_1d)
#         x_v = self.conv_v(x_1d)        # n, c, t
#
#         # attention = torch.matmul(x_q, x_k.transpose(1, 2).contiguous())   # n, c, c
#         attention = (x_q @ x_k.transpose(1, 2)) * (h ** -1)
#
#         # x_change = torch.matmul(x_change, x_change.transpose(1, 2).contiguous())
#         x_change = (x_change @ x_change.transpose(1, 2)) * (h ** -1)
#
#         x_out = x_change + attention
#         # x_out = x_out.softmax(dim=0)
#
#         x_out = torch.matmul(x_out, x_v)   # n , c, t
#
#         x_out = self.conv_after(x_out)
#
#         x_out = x_out.permute(0, 2, 1).reshape(n_batch, args.clip_len, c, 1, 1).contiguous()
#         x_out = x_out.reshape(nt, c, 1, 1).contiguous()
#         # x_out = self.sigmoid(x_out)
#         # x_out = self.relu(x_out)
#
#         return x_out   #nt, c, 1, 1




class My1DBlock(nn.Module):
    def __init__(self, channel):
        super(My1DBlock, self).__init__()
        self.in_channel = channel
        self.out_channel = self.in_channel // args.reduce_num

        self.q = nn.Linear(self.in_channel, self.out_channel, bias=False)
        self.k = nn.Linear(self.in_channel, self.out_channel, bias=False)
        self.v = nn.Linear(self.in_channel, self.out_channel, bias=False)

        self.restore = nn.Linear(self.out_channel, self.in_channel, bias=False)

        # self.conv_q = nn.Conv1d(self.in_channel, self.out_channel , kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        # self.conv_k = nn.Conv1d(self.in_channel, self.out_channel, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        # self.conv_v = nn.Conv1d(self.in_channel, self.out_channel, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.conv_before = nn.Conv1d(self.in_channel, self.out_channel, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.conv_after = nn.Conv1d(self.out_channel, self.in_channel, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pad = (0, 0, 1, 0)
        self.positionalEncoding1D = PositionalEncoding1D(self.in_channel)   # 1D ： [Batch size, Sequence length, Channels]


    def forward(self, x):
        x_1d = self.avg_pool(x)   # nt, c, 1, 1
        nt, c, h, w = x_1d.size()
        n_batch = nt // args.clip_len

        x_1d = x_1d.reshape(nt, c).reshape(n_batch, args.clip_len, c).contiguous()  # n, t, c
        # 1D ： [Batch size, Sequence length, Channels]
        x_1d = self.positionalEncoding1D(x_1d)
        x_1d = x_1d.to(x.device)    #n, t, c

        x3_plus0, _ = x_1d.split([args.clip_len - 1, 1], dim=1)
        _, x3_plus1 = x_1d.split([1, args.clip_len - 1], dim=1)
        x_p3 = x3_plus1 - x3_plus0                       # n,t,c
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)   # n, t, c

        x_q = self.q(x_p3)
        x_k = self.k(x_p3)
        x_v = self.v(x_1d)

        attention = (x_q @ x_k.permute(0, 2, 1)) * (c ** -0.5)
        attention = attention.softmax(dim=1)
        x_out = (attention @ x_v)
        x_out = self.restore(x_out)  # n, t, c
        x_out = x_out.reshape(nt, c, 1, 1)
        return x_out

