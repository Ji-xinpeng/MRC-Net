class MyChannelBlock(nn.Module):
    def __init__(self, inter_channel):
        super(MyChannelBlock, self).__init__()
        self.inter_channel = inter_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_aaten = attention.Attention(args.clip_len, num_heads=args.num_heads, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.positionalEncoding1D = PositionalEncoding1D(self.inter_channel)  # 1D ： [Batch size, Sequence length, Channels]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        x_p2 = self.avg_pool(x)
        nt, c, h, w = x_p2.size()
        n_batch = nt // args.clip_len
        x_p2 = x_p2.reshape(n_batch, args.clip_len, c, h, w)
        x_p2 = x_p2.reshape(n_batch, args.clip_len, c).contiguous()    # n, t, c
        x_p2 = self.positionalEncoding1D(x_p2)
        x_p2 = x_p2.to(x.device)
        x_p2 = x_p2.permute(0, 2, 1)
        x_change = torch.roll(x_p2, shifts=args.begin_split, dims=2)
        x_change = x_change + x_p2
        x = x_out
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.to(x.device)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x_out = x
        x_out = x_out.permute(0, 2, 1).view(n_batch, args.clip_len, c, 1, 1)
        x_out = x_out.reshape(nt, c, h, w).contiguous()
        return x_out

