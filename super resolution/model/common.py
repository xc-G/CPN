import math
from .MTLU_Package.MTLU import MTLU
from .MTLU_Package.MTLU import MTSiLU
from .MTLU_Package.MTLU import MTLU_continuous
from timm.models.layers import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def sigma(x):
    return 1/(1 + math.exp(-x))


class CPN_nl(nn.Module):
    def __init__(self, FeatMapNum=64,BinNum=2):
        super(CPN_nl,self).__init__()
        self.BinNum = BinNum
        self.FeatMapNum = FeatMapNum
        self.coef = nn.Parameter(torch.zeros(FeatMapNum, 3*BinNum))
        HalfBinNum = int(BinNum/2)
        self.coef.data[:,:HalfBinNum] = 1
        self.silu = nn.SiLU()
 
    def forward(self, x):
        a = self.coef[:, :self.BinNum].reshape(1,self.FeatMapNum,self.BinNum)
        b = self.coef[:, self.BinNum : 2 * self.BinNum].reshape(1,self.FeatMapNum,self.BinNum)
        c = self.coef[:, 2 * self.BinNum : 3 * self.BinNum].reshape(1,self.FeatMapNum,self.BinNum)

        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        output = a * x_perm + b * self.silu(x_perm) + c
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1) 
        return result

class CPN_nl_2(nn.Module):
    def __init__(self, FeatMapNum=64):
        super(CPN_nl_2,self).__init__()
        self.y1 = nn.Parameter(torch.zeros(1,FeatMapNum,1,1))
        self.y2 = nn.Parameter(torch.ones(1,FeatMapNum,1,1))
        self.silu = nn.SiLU()

    def forward(self, x):
        self.c = self.y1
        self.b = (self.y2 - 2 * self.y1)/(sigma(1)-sigma(-1))
        self.a = self.y1 - sigma(-1) * (self.y2 - 2 * self.y1)/(sigma(1)-sigma(-1))

        x = self.a * x + self.b * self.silu(x) + self.c
        return x

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)

class CPN_nl_1(nn.Module):
    def __init__(self, n_feats = 64):
        super(CPN_nl_1, self).__init__()
        self.mtlu = MTLU(FeatMapNum=n_feats)
        self.mtsilu = MTSiLU(FeatMapNum=n_feats)

    def forward(self, x):
        mtlu_x = self.mtlu(x)
        mtsilu_x = self.mtsilu(x)
        return mtlu_x + mtsilu_x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=True, res_scale=1):
        if act:
            act = nn.ReLU(True)
        else:
            act = CPN_nl(FeatMapNum=n_feats,BinNum=4)
            # act = CPN_nl_1(n_feats=n_feats)
            # act = CPN_nl_2(FeatMapNum=n_feats)
            # act = MTLU(FeatMapNum=n_feats)

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))

            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ChannelAttention_(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention_(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, kernel_size = 7, inplanes=64, ratio=16):
        super(CBAM,self).__init__()
        self.ca = ChannelAttention_(inplanes,ratio)
        self.sa = SpatialAttention_(kernel_size)

    def forward(self,x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class pixelattention(nn.Module):
    def __init__(self, kernel_size = 7, inplanes=64, ratio=16):
        super(pixelattention,self).__init__()
        self.relu = nn.ReLU(True)
        self.ca = ChannelAttention_(inplanes,ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):

        ca_out = self.ca(x)
        sa_out = self.sa(x)
        x = x*ca_out*sa_out
        return x


class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=4):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        a = self.conv1(a)
        a = self.sigmoid(a)
        return x*a

class CAB(nn.Module):

    def __init__(self, num_feats, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feats, num_feats // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feats // compress_ratio, num_feats, 3, 1, 1),
            ChannelAttention(num_feats, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class body(nn.Module):
    def __init__(self, depth, stage=2, stage_length = 3, conv = default_conv,n_feats=64, kernel_size=3, act=True, res_scale=1,compress_ratio=4,squeeze_factor=4, num_heads=4):
        super().__init__()
        self.stage_length = stage_length
        self.depth = depth
        self.stage = stage
        self.convs1 = nn.ModuleList([
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for i in range(stage_length)
        ])
        self.convs2 = nn.ModuleList([
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for i in range(stage_length)
        ])
        self.hbs1 = nn.ModuleList([
            HybridBlock(
                n_feats,compress_ratio,squeeze_factor, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=True) for i in range(stage_length)
        ])
        self.hbs2 = nn.ModuleList([
            HybridBlock(
                n_feats,compress_ratio,squeeze_factor, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=True) for i in range(stage_length)
        ])

    def forward(self, x):
        shortcut = x
        for conv in self.convs1:
            x = conv(x)
        x = shortcut + x

        shortcut = x
        for conv in self.convs2:
            x = conv(x)
        x = shortcut + x

        shortcut = x
        for hb in self.hbs1:
            x = hb(x)
        x = shortcut + x        

        shortcut = x
        for hb in self.hbs2:
            x = hb(x)
        x = shortcut + x    

        return x








class HybridBlock(nn.Module):

    def __init__(self, n_feats,compress_ratio,squeeze_factor, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=True):
        super().__init__()
        self.norm1 = norm_layer(n_feats)
        self.attn = Attention(
            n_feats,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(n_feats)
        mlp_hidden_dim = int(n_feats * mlp_ratio)
        self.mlp = Mlp(in_features=n_feats, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        
        self.conv_block = CAB(num_feats=n_feats, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)



    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x = self.norm1(x)
        conv_x = self.conv_block(x)
        conv_x = conv_x.permute(0,3,2,1).contiguous.view(b,h*w,c)

        attn_x = self.attn(x,h,w)
        x = shortcut + self.drop_path(attn_x) + conv_x * 0.01

        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, n_feats=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True, groups=n_feats)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)


    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

