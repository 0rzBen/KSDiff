import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops.layers.torch import Rearrange
from models.DyConv import (PRConv2d,PRConvNeXtBlock, PRConvTranspose2d, 
                    PRConv2dLite, PRConvNeXtBlockLite, PRConvTranspose2dLite, PRConvNeXtBlockScale)

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)



class ConvNeXtBlock(nn.Module):
    """
    A single ConvNeXt Block (DwConv + LN + Conv) with residual connection.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)).view(1, dim, 1, 1))
        else:
            self.gamma = None

    def forward(self, x):  # x: B x C x H x W
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return input + x

class PRConvNeXtBlockSequence(nn.Module):
    def __init__(self, dim, layer_scale_init, depth, embed_dim=64):
        super().__init__()
        self.blocks = nn.ModuleList([PRConvNeXtBlockScale(dim, embed_dim=embed_dim, layer_scale_init_value=layer_scale_init,depth=depth) for _ in range(depth)])
        
    def forward(self, x, prior):
        for block in self.blocks:
            x = block(x, prior)
        return x, prior

class FusionNetConvNeXtScale(nn.Module):
    """
    Multi-scale pansharpening network using ConvNeXt blocks.
    Basic pansharpening operations (pan-lms diff, add back) are preserved.
    U-shaped encoder-decoder with learnable linear interpolation for skip connections.
    """
    def __init__(self, spectral_num, base_dim=32, depths=[1, 2, 2, 4], layer_scale_init=1e-6, embed_dim=None, group=4,): #default
        super().__init__()
        self.spectral_num = spectral_num
        self.depths = depths

        dims = [base_dim * m for m in depths]

        # Initial embedding: from spectral_num channels to dims[0]
        self.stem = nn.Conv2d(spectral_num, dims[0], kernel_size=3, stride=1, padding=1, bias=True)

        if embed_dim is None:
            # Encoder
            self.encoder_blocks = nn.ModuleList([
                nn.Sequential(*[ConvNeXtBlock(d, layer_scale_init) for _ in range(depths[i])])
                for i, d in enumerate(dims)
            ])
            self.downsamples = nn.ModuleList([
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                for i in range(len(dims) - 1)
            ])

            # Decoder
            self.upsamples = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()
            self.skip_alphas = nn.ParameterList()
            # Build decoder in reverse scale order
            for i in range(len(dims) - 1, 0, -1):
                # Upsample from dims[i] -> dims[i-1]
                self.upsamples.append(
                    nn.ConvTranspose2d(dims[i], dims[i-1], kernel_size=2, stride=2)
                )
                # ConvNeXt refinement after skip
                self.decoder_blocks.append(
                    nn.Sequential(*[ConvNeXtBlock(dims[i-1], layer_scale_init) for _ in range(depths[i-1])])
                )
                # Learnable interpolation weight per channel, init=0.5
                alpha = nn.Parameter(torch.ones(1, dims[i-1], 1, 1) * 0.5)
                self.skip_alphas.append(alpha)
        else:
            self.down_1 = nn.Sequential(
                Rearrange('b n c -> b c n'),
                nn.Linear(group*group, (group*group)//4),
                Rearrange('b c n -> b n c'),
                nn.Linear(embed_dim*4, embed_dim*4)
            )
            self.down_2 = nn.Sequential(
                Rearrange('b n c -> b c n'),
                nn.Linear((group*group)//4, 1),
                Rearrange('b c n -> b n c'),
                nn.Linear(embed_dim*4, embed_dim*4)
            )

            
            self.norm = LayerNorm_Without_Shape(embed_dim*4, BiasFree_LayerNorm)
            self.ln = nn.ModuleList([nn.Linear(embed_dim*4, d*base_dim) for d in depths])
            # Encoder
            self.encoder_blocks = nn.ModuleList([
                PRConvNeXtBlockSequence(d, layer_scale_init, depths[i], embed_dim=embed_dim) 
                for i, d in enumerate(dims)
            ])
            self.downsamples = nn.ModuleList([
                PRConv2dLite(dims[i], dims[i+1], kernel_size=2, stride=2, rank_group=(dims[i]//4, dims[i+1]//4, 1, 1))
                for i in range(len(dims) - 1)
            ])

            # Decoder
            self.upsamples = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()
            self.skip_alphas = nn.ParameterList()
            # Build decoder in reverse scale order
            for i in range(len(dims) - 1, 0, -1):
                # Upsample from dims[i] -> dims[i-1]
                self.upsamples.append(
                    PRConvTranspose2dLite(dims[i], dims[i-1], kernel_size=2, stride=2)
                )
                # ConvNeXt refinement after skip
                self.decoder_blocks.append(
                    PRConvNeXtBlockSequence(dims[i-1], layer_scale_init, depths[i-1], embed_dim=embed_dim)
                )
                # Learnable interpolation weight per channel, init=0.5
                alpha = nn.Parameter(torch.ones(1, dims[i-1], 1, 1) * 0.5)
                self.skip_alphas.append(alpha)

        # Final projection to spectral_num
        self.conv_out = nn.Conv2d(dims[0], spectral_num, kernel_size=3, stride=1, padding=1, bias=True)

    def cast_prior(self, z: torch.Tensor, idx: int) -> torch.Tensor:

        assert z.dim() == 3, "Input z must have shape (batch_size, N, in_channels)"
        b, N, c_in = z.shape
        
        z = self.norm(z)
        z = self.ln[idx](z)

        return z

    def forward(self, lms, pan, prior=None, learn_res=False):
        # lms: low-res multispectral (B x spectral_num x H x W)
        # pan: high-res panchromatic (B x 1 x H x W)

        # Basic pansharpening diff
        pan_concat = pan.repeat(1, self.spectral_num, 1, 1)
        x = pan_concat - lms
        

        # Stem
        x = self.stem(x)

        if prior is None:
            # Encoder
            skips = []
            for i, block in enumerate(self.encoder_blocks):
                x = block(x)
                skips.append(x)
                if i < len(self.downsamples):
                    x = self.downsamples[i](x)

            # Decoder with learnable linear interpolation
            for idx in range(len(self.upsamples)):
                x = self.upsamples[idx](x)
                skip = skips[-2 - idx]
                # Compute interpolation weight
                alpha = torch.sigmoid(self.skip_alphas[idx])
                # Interpolate between x and skip
                x = alpha * x + (1 - alpha) * skip
                # Refinement
                x = self.decoder_blocks[idx](x)
        else:
            # Encoder
            prior_1 = prior
            prior_2 = self.down_1(prior_1)
            prior_3 = self.down_2(prior_2) # multi-scale for N

            skips = []
            for i, block in enumerate(self.encoder_blocks):

                if self.depths[i] == 1:
                    prior = prior_1
                elif self.depths[i] == 2:
                    prior = prior_2
                elif self.depths[i] == 4:
                    prior = prior_3
                
                x, _ = block(x, prior)

                skips.append(x)
                if i < len(self.downsamples):
                    if x.shape[1] != prior.shape[2]:
                        x = self.downsamples[i](x, self.cast_prior(prior, i)) 
                    else:
                        x = self.downsamples[i](x, prior)

            # Decoder with learnable linear interpolation
            for idx in range(len(self.upsamples)):
                if x.shape[1] != prior.shape[2]:
                    x = self.upsamples[idx](x, self.cast_prior(prior, len(self.depths)-1-idx))
                else:
                    x = self.upsamples[idx](x, prior)
                skip = skips[-2 - idx]
                # Compute interpolation weight
                alpha = torch.sigmoid(self.skip_alphas[idx])
                # Interpolate between x and skip
                x = alpha * x + (1 - alpha) * skip
                # Refinement
                x, _ = self.decoder_blocks[idx](x, prior)

        # Output and add back low-res multispectral
        out = self.conv_out(x)
        if not learn_res:
            return out + lms
        else:
            return out


if __name__ == "__main__":
    image_size = 64
    batch_size = 64

    lms = torch.randn(batch_size, 8, image_size, image_size)
    pan = torch.randn(batch_size, 1, image_size, image_size)

    prior = torch.randn(batch_size, 16, 256)


    model = FusionNetConvNeXtScale(8, base_dim=64, depths=[1, 2, 2, 4], 
                                                layer_scale_init=1e-6, embed_dim=64)

    output = model(lms, pan=pan, prior=prior)
    


