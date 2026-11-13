import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY

class MLP(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 **kwargs):

        super(MLP, self).__init__()

        patch_mix_dims = int(patch_expansion * num_patches)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))

        return x


@ARCH_REGISTRY.register()
class mid_denoiser(nn.Module):
    def __init__(
        self,
        in_channel=256, # (embed_dim*4)
        out_channel=256, # (embed_dim*4)
        inner_channel=512,
        block_num=6,
        n_steps=25,
        group=4, 
        patch_expansion=0.5, 
        channel_expansion=2
    ):
        super().__init__()

        self.time_mlp = nn.Parameter((1 / n_steps) * torch.ones(1), requires_grad=True)
        self.self_condition = None
        self.pred_var = False

        self.first_layer = nn.Sequential(
                nn.Linear(in_channel*3, inner_channel),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.blocks = nn.ModuleList()
        for i in range(block_num-2):
            block = nn.Sequential(
                MLP(num_patches=group*group, embed_dims=inner_channel, patch_expansion=patch_expansion, channel_expansion=channel_expansion),
                nn.GELU(),)
            self.blocks.append(block)

        self.final_layer = nn.Sequential(
                nn.Linear(inner_channel, out_channel),
                nn.GELU(),)

    def forward(self, x, time, cond=None, self_cond=None):
        b, n, c = x.shape

        t = time * self.time_mlp
        t = t.view(b, 1, 1).repeat(1, n, c)

        x = torch.cat([x, t, cond], dim=-1)
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x)

if __name__ == "__main__":
    b, n, c = 4, 16, 256
    x = torch.randn(b, n, c)
    time = torch.randint(0, 25, (b,), device=x.device).long()
    cond = torch.randn(b, n, c)

    model = mid_denoiser( in_channel=256, # (embed_dim*4)
                        out_channel=256, # (embed_dim*4)
                        inner_channel=512,
                        block_num=4,
                        n_steps=25,
                        group=4, 
                        patch_expansion=0.5, 
                        channel_expansion=2)
    output = model(x, time, cond=cond)

    print("output: ", output.shape) #output:  torch.Size([b, n, c])
