from others.params import *
from my_moudle.local_attention import *
from positional_encodings.torch_encodings import PositionalEncoding1D


class MyChannelBlock(nn.Module):
    def __init__(self, inter_channel):
        super(MyChannelBlock, self).__init__()
        self.inter_channel = inter_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_aaten = attention.Attention(args.clip_len, num_heads=args.num_heads, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.positionalEncoding1D = PositionalEncoding1D(self.inter_channel)  # 1D ： [Batch size, Sequence length, Channels]

    def forward(self, x):
        x_p2 = self.avg_pool(x)
        nt, c, h, w = x_p2.size()
        n_batch = nt // args.clip_len

        x_p2 = x_p2.reshape(n_batch, args.clip_len, c, h, w)
        x_p2 = x_p2.reshape(n_batch, args.clip_len, c).contiguous()    # n, t, c

        x_p2 = self.positionalEncoding1D(x_p2)
        x_p2 = x_p2.to(x.device)
        x_p2 = x_p2.permute(0, 2, 1)  # n , c , t

        # x_change = torch.roll(x_p2, shifts=args.begin_split, dims=2)
        # x_change = x_change + x_p2

        x_out = self.channel_aaten(x_p2)

        x_out = x_out.permute(0, 2, 1).view(n_batch, args.clip_len, c, 1, 1)
        x_out = x_out.reshape(nt, c, h, w).contiguous()
        # x_out = self.sigmoid(x_out)
        # x_out = self.relu(x_out)

        return x_out
