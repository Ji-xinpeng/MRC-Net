
from others.params import *
from my_moudle.local_attention import *
from positional_encodings.torch_encodings import PositionalEncoding2D, PositionalEncoding3D, PositionalEncoding1D


class My2DBlock(nn.Module):
    def __init__(self, channel):
        super(My2DBlock, self).__init__()
        self.in_channel = channel
        self.inter_channel = channel // args.reduce_num
        # mobilenet zhili must be 1
        self.inter_channel = 1
        self.in_channel = 1
        self.conv_q = nn.Conv2d(in_channels=1, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_k = nn.Conv2d(in_channels=1, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=self.in_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.positionalEncoding2D = PositionalEncoding2D(1)  # 2D :  [Batch size, Height, Width, Channels]
        self.positionalEncoding3D = PositionalEncoding3D(1)
        self.positionalEncoding1D = PositionalEncoding1D(1)


    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // args.clip_len

        # positionalEncoding2D
        x_2d = x.permute(0, 2, 3, 1)   # nt, h, w, c
        x_2d = x_2d.mean(3, keepdim=True).contiguous()
        x_2d = x_2d.reshape(n_batch, args.clip_len, h, w, 1)
        x_2d = self.positionalEncoding3D(x_2d)


        #---------------------------------------------------------------------------------#
        # x_2d = x_2d.reshape(n_batch, args.clip_len*h*w, 1)
        # x_2d = self.positionalEncoding1D(x_2d)
        # x_2d = x_2d.reshape(n_batch, args.clip_len, h, w, 1)
        # ---------------------------------------------------------------------------------#



        x_2d = x_2d.to(x.device)
        x_2d = x_2d.permute(0, 1, 4, 2, 3).contiguous()  # n,t,1,h,w
        c = 1
        x3_plus0, _ = x_2d.view(n_batch, args.clip_len, c, h, w).split([args.clip_len - 1, 1], dim=1)
        _, x3_plus1 = x_2d.view(n_batch, args.clip_len, c, h, w).split([1, args.clip_len - 1], dim=1)
        x_p3 = x3_plus1 - x3_plus0
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3 = x_p3.reshape(nt, c, h, w).contiguous()

        x_q = self.conv_q(x_p3)
        x_k = self.conv_k(x_p3)
        x_2d = x_2d.reshape(nt, c, h, w).contiguous()
        x_v = self.conv_v(x_2d)

        attention = (x_q @ x_k.permute(0, 1, 3, 2))  # n, c, h, w
        attention = attention.softmax(dim=1)

        x_out = (attention @ x_v)
        x_out = self.conv_mask(x_out)
        # x_out = self.sigmoid(x_out)

        return x_out   # nt, c, h, w

# if __name__=='__main__':
#     model = My2DBlock(64)
#
#     input = torch.randn(16, 64, 56, 56)
#     out = model(input)
#     print(out.shape)


