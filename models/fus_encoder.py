import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class FastLinearCrossAttention(nn.Module):
    """Lightweight linear cross-modal attention, configurable number of heads"""
    def __init__(self, fea_dim, cond_dim, nheads=2):
        super().__init__()
        assert fea_dim % nheads == 0, "fea_dim must be divisible by nheads"
        self.nheads = nheads
        self.head_dim = fea_dim // nheads
        self.scale = 1 / math.sqrt(self.head_dim)

        # 1x1 projection
        self.q = nn.Conv2d(fea_dim, fea_dim, 1, bias=False)
        self.kv = nn.Conv2d(cond_dim, fea_dim * 2, 1, bias=False)
        self.proj = nn.Conv2d(fea_dim, fea_dim, 1)

    def forward(self, x, cond):
        b, c, h, w = x.shape
        q = self.q(x)
        kv = self.kv(cond)
        k, v = kv.chunk(2, dim=1)

        # Multi-head reshape
        q = rearrange(q, 'b (h c) X Y -> b h c (X Y)', h=self.nheads)
        k = rearrange(k, 'b (h c) X Y -> b h c (X Y)', h=self.nheads)
        v = rearrange(v, 'b (h c) X Y -> b h c (X Y)', h=self.nheads)

        # Normalization
        q = q.softmax(dim=-1) * self.scale
        k = k.softmax(dim=-2)

        # Linear attention computation
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (X Y) -> b (h c) X Y', X=h, Y=w)

        return self.proj(out)

class DynamicFusionGate(nn.Module):
    """Fusion gate customized based on channels of x and cond"""
    def __init__(self, dim_x, dim_cond):
        super().__init__()
        self.gate = nn.Conv2d(dim_x + dim_cond, dim_x, 1)

    def forward(self, x1, x2, inter):
        g = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=1)))
        return g * x1 + (1 - g) * x2 + inter

class StemBlock(nn.Module):
    """Lightweight StemBlock"""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
    def forward(self, x):
        return F.relu(self.proj(x), inplace=True)

class CrossModalStage(nn.Module):
    """Lightweight interaction stage, cond_dim fixed as embed channel"""
    def __init__(self, dim, cond_dim, nheads=2):
        super().__init__()
        self.attn = FastLinearCrossAttention(dim, cond_dim, nheads)
        self.cond_proj = nn.Conv2d(cond_dim, dim, 1)
        self.gate = DynamicFusionGate(dim, dim)
        self.down = nn.Conv2d(dim, dim * 2, 2, stride=2)

    def forward(self, x, cond): # p, g
        inter = self.attn(x, cond)
        aligned = self.cond_proj(cond)
        fused = self.gate(x, aligned, inter)
        return self.down(fused)

class PLFE_Stage1(nn.Module):
    """Stage 1: Input pan, lms, gt, output [B, N, C]"""
    def __init__(
        self,
        pan_c=1,
        lms_c=8,
        gt_c=8,
        base_dim=32,
        num_stages=3,
        nheads=8,
        group=4,
        kernel_size=3
    ):
        super().__init__()
        self.group = group
        # Initial embedding
        self.pan_embed = StemBlock(pan_c, base_dim, kernel_size)
        self.lms_embed = StemBlock(lms_c, base_dim, kernel_size)
        self.gt_embed = StemBlock(gt_c, base_dim, kernel_size)
        self.final_dim = base_dim * 4
        # Pyramid stages: Interaction of pan and lms with gt
        self.pan_stages = nn.ModuleList()
        self.lms_stages = nn.ModuleList()
        dims = [base_dim * (2 ** i) for i in range(num_stages + 1)]
        for i in range(num_stages):
            self.pan_stages.append(
                CrossModalStage(dim=dims[i], cond_dim=base_dim, nheads=nheads)
            )
            self.lms_stages.append(
                CrossModalStage(dim=dims[i], cond_dim=base_dim, nheads=nheads)
            )
        # Output convolution: Concatenate final stage pan, lms, gt
        in_ch = dims[-1] * 2 + base_dim
        self.conv = nn.Conv2d(in_ch, self.final_dim, 1) # ensure
        self.pool = nn.AdaptiveAvgPool2d((group, group))

    def forward(self, lms=None, pan=None, gt=None, cond=None, learn_res=False):
        p = self.pan_embed(pan)
        l = self.lms_embed(lms)
        g = self.gt_embed(gt)
        # Multi-scale interaction
        for pan_stage, lms_stage in zip(self.pan_stages, self.lms_stages):
            p = pan_stage(p, g)
            l = lms_stage(l, g)
            g = F.interpolate(g, scale_factor=0.5, mode='bilinear', align_corners=False)
        # Concatenation
        feat = torch.cat([p, l, g], dim=1)
        feat = F.relu(self.conv(feat), inplace=True)
        feat = self.pool(feat)
        b, c, h, w = feat.shape
        return rearrange(feat, 'b c h w -> b (h w) c')

class PLFE_Stage2(nn.Module):
    """Stage 2: Input pan, lms, output [B, N, C]"""
    def __init__(
        self,
        pan_c=1,
        lms_c=8,
        base_dim=32,
        num_stages=3,
        nheads=8,
        group=4,
        kernel_size=3
    ):
        super().__init__()
        self.group = group
        self.pan_embed = StemBlock(pan_c, base_dim, kernel_size)
        self.lms_embed = StemBlock(lms_c, base_dim, kernel_size)
        self.final_dim = base_dim * 4
        self.stages = nn.ModuleList()
        dims = [base_dim * (2 ** i) for i in range(num_stages + 1)]
        for i in range(num_stages):
            self.stages.append(
                CrossModalStage(dim=dims[i], cond_dim=base_dim, nheads=nheads)
            )
        in_ch = dims[-1] + base_dim
        self.conv = nn.Conv2d(in_ch, self.final_dim, 1) # ensure 
        self.pool = nn.AdaptiveAvgPool2d((group, group))

    def forward(self, lms=None, pan=None, cond=None, learn_res=False):
        p = self.pan_embed(pan)
        l = self.lms_embed(lms)
        for stage in self.stages:
            p = stage(p, l)
            l = F.interpolate(l, scale_factor=0.5, mode='bilinear', align_corners=False)
        feat = torch.cat([p, l], dim=1)
        feat = F.relu(self.conv(feat), inplace=True)
        feat = self.pool(feat)
        b, c, h, w = feat.shape
        return rearrange(feat, 'b c h w -> b (h w) c')



if __name__ == "__main__":
    pan = torch.randn(64, 1, 64, 64)
    lms = torch.randn(64, 8, 64, 64)
    gt = torch.randn(64, 8, 64, 64)

    encoder = HMCFEStage1Light(pan_c=1, lms_c=8, gt_c=8, base_dim=64, group=4, num_stages=3)
    latent = encoder(pan=pan, lms=lms, gt=gt)


   

    