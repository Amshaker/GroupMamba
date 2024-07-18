import torch
import torch.nn as nn
from functools import partial
import torch.fft

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

from einops import rearrange

try:
    from .ss2d import SS2D
    from .csms6s import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
    from .csms6s import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4
except:
    from ss2d import SS2D
    from csms6s import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
    from csms6s import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GroupMambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=1, d_conv=3, expand=1, reduction=16):
        super().__init__()

        num_channels_reduced = input_dim // reduction
        self.fc1 = nn.Linear(input_dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba_g1 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g2 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g3 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g4 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, H, W):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, N, C = x.shape
        x = self.norm(x)

        # Channel Affinity
        z = x.permute(0, 2, 1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        x = rearrange(x, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)

        # Four scans applied to 4 different directions, each is applied for N/4 channels
        x_mamba1 = self.mamba_g1(x1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1)
        x_mamba2 = self.mamba_g2(x2, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2)
        x_mamba3 = self.mamba_g3(x3, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3)
        x_mamba4 = self.mamba_g4(x4, CrossScan=CrossScan_4, CrossMerge=CrossMerge_4)

        # Combine all feature maps
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1) * self.skip_scale * x

        x_mamba = rearrange(x_mamba, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)

        # Channel Modulation
        x_mamba = x_mamba * fc_out_2.unsqueeze(1)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)

        return x_mamba

class ClassBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = GroupMambaLayer(dim, dim)
        self.mlp = FFN(dim, int(dim * mlp_ratio))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.norm1(self.attn(x[:, :1], H, W))
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed), H, W)
        return torch.cat([cls_embed, x[:, 1:]], dim=1)

class Block_mamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)

        self.attn = GroupMambaLayer(dim, dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(x, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class GroupMamba(nn.Module):
    def __init__(self, 
        in_chans=3, 
        num_classes=1000, 
        stem_hidden_dim = 32,
        embed_dims=[64, 128, 348, 448],
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        num_stages=4,
        distillation=True,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])
            block = nn.ModuleList([Block_mamba(
                    dim = embed_dims[i],
                    mlp_ratio = mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer)
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        post_layers = ['ca']
        self.post_network = nn.ModuleList([
            ClassBlock(
                dim = embed_dims[-1], 
                mlp_ratio = mlp_ratios[-1],
                norm_layer=norm_layer)
            for _ in range(len(post_layers))
        ])

        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # distillation head
        self.dist = distillation
        if self.dist:
            self.dist_head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_cls(self, x, H, W):
        cls_tokens = x.mean(dim=1, keepdim=True)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x, H, W)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x, 1, 1)[:, 0]
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x

    def forward(self, x):

        x = self.forward_features(x)
        if self.dist:
            cls_out = self.head(x), self.dist_head(x)
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x)

        return cls_out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

@register_model
def groupmamba_tiny(pretrained=False, **kwargs):
    model = GroupMamba(
        stem_hidden_dim = 32,
        embed_dims = [64, 128, 348, 448], 
        mlp_ratios = [8, 8, 4, 4],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 4, 9, 3],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def groupmamba_small(pretrained=False, **kwargs):
    model = GroupMamba(
        stem_hidden_dim = 64,
        embed_dims = [64, 128, 348, 512], 
        mlp_ratios = [8, 8, 4, 4], 
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 4, 16, 3],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def groupmamba_base(pretrained=False, **kwargs):
    model = GroupMamba(
        stem_hidden_dim = 64,
        embed_dims = [96, 192, 424, 512],
        mlp_ratios = [8, 8, 4, 4],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 6, 21, 3],
        **kwargs)
    model.default_cfg = _cfg()
    return model