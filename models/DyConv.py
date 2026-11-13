import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numbers
import math


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


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

class PResblock(nn.Module):
    def __init__(self):
        super(PResblock, self).__init__()

        channel = 32
        self.conv20 = PRConv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True, rank_group=(16, 16, 1, 1))
        self.conv21 = PRConv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True, rank_group=(16, 16, 1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prior=None):  
        rs1 = self.relu(self.conv20(x, prior))  
        rs1 = self.conv21(rs1, prior)  
        rs = torch.add(x, rs1) 
        return rs

class PResblockLite(nn.Module):
    def __init__(self, rank_group=(8, 8, 1, 1), channel=32):
        super(PResblockLite, self).__init__()

         
        self.conv20 = PRConv2dLite(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True, rank_group=rank_group)
        self.conv21 = PRConv2dLite(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True, rank_group=rank_group)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prior=None):  
        rs1 = self.relu(self.conv20(x, prior))  
        rs1 = self.conv21(rs1, prior)  
        rs = torch.add(x, rs1) 
        return rs

class PRConvNeXtBlock(nn.Module):
    """
    A single ConvNeXt Block (DwConv + LN + Conv) with residual connection.
    """
    def __init__(self, dim, embed_dim=64, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = PRConv2d(dim, dim, kernel_size=7, padding=3, groups=dim, rank_group=(dim//4, dim//4, 3, 3))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = PRConv2d(dim, 4 * dim, kernel_size=1, rank_group=(dim//4, dim, 1, 1))
        self.act = nn.GELU()
        self.pwconv2 = PRConv2d(4 * dim, dim, kernel_size=1, rank_group=(dim, dim//4, 1, 1))
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)).view(1, dim, 1, 1))
        else:
            self.gamma = None
        
        self.norm_z = LayerNorm_Without_Shape(embed_dim*4, BiasFree_LayerNorm)
        self.ln_z = nn.Linear(embed_dim*4, dim)
        self.ln_z2 = nn.Linear(embed_dim*4, dim*4) # for self.pwconv2

    def cast_prior(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 3, "Input z must have shape (batch_size, N, in_channels)"
        b, N, c_in = z.shape
        z = self.norm_z(z)
        z = self.ln_z(z)
        return z

    def cast_prior_2(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 3, "Input z must have shape (batch_size, N, in_channels)"
        b, N, c_in = z.shape
        z = self.norm_z(z)
        z = self.ln_z2(z)
        return z
    
    def forward(self, x, prior=None):  # x: B x C x H x W
        input = x
        
        x = self.dwconv(x, self.cast_prior(prior))
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        x = self.pwconv1(x, self.cast_prior(prior))
        x = self.act(x)
        
        x = self.pwconv2(x, self.cast_prior_2(prior))
        if self.gamma is not None:
            x = self.gamma * x
        return input + x

class PRConvNeXtBlockLite(nn.Module):
    """
    A single ConvNeXt Block (DwConv + LN + Conv) with residual connection.
    """
    def __init__(self, dim, embed_dim=64, layer_scale_init_value=1e-6, resizer=(8, 16, 8, 16)):
        super().__init__()

        # model size controller
        min1, max1, min2, max2 = resizer
        rank_dim = min(max(dim, min1), max1)
        rank_4xdim = min(max(dim, min2), max2)

        self.dwconv = PRConv2dLite(dim, dim, kernel_size=7, padding=3, groups=dim, rank_group=(rank_dim, rank_dim, 3, 3))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = PRConv2dLite(dim, 4 * dim, kernel_size=1, rank_group=(rank_dim, rank_4xdim, 1, 1))
        self.act = nn.GELU()
        self.pwconv2 = PRConv2dLite(4 * dim, dim, kernel_size=1, rank_group=(rank_4xdim, rank_dim, 1, 1))
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)).view(1, dim, 1, 1))
        else:
            self.gamma = None
        
        self.norm_z = LayerNorm_Without_Shape(embed_dim*4, BiasFree_LayerNorm)
        self.ln_z = nn.Linear(embed_dim*4, dim)
        self.ln_z2 = nn.Linear(embed_dim*4, dim*4) # for self.pwconv2

    def cast_prior(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 3, "Input z must have shape (batch_size, N, in_channels)"
        b, N, c_in = z.shape
        z = self.norm_z(z)
        z = self.ln_z(z)
        return z

    def cast_prior_2(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 3, "Input z must have shape (batch_size, N, in_channels)"
        b, N, c_in = z.shape
        z = self.norm_z(z)
        z = self.ln_z2(z)
        return z
    
    def forward(self, x, prior=None):  # x: B x C x H x W
        input = x
        
        x = self.dwconv(x, self.cast_prior(prior))
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B C H W
        x = self.pwconv1(x, self.cast_prior(prior))
        x = self.act(x)
        
        x = self.pwconv2(x, self.cast_prior_2(prior))
        if self.gamma is not None:
            x = self.gamma * x
        return input + x

class PRConvNeXtBlockScale(nn.Module):
    """
    A single ConvNeXt Block (DwConv + LN + Conv) with residual connection.
    """
    def __init__(self, dim, embed_dim=64, layer_scale_init_value=1e-6, depth=1, dim_resizer=(8, 16, 8, 16)):
        super().__init__()
        # default:embed_dim=64, layer_scale_init_value=1e-6, depth=1, dim_resizer=(4, 8, 4, 16)
        
        # model size controller
        kernel_scale = int(-4*math.log2(depth) + 11) # [1, 2, 2, 4] -> [11, 7, 7, 3]
        rank_k = (kernel_scale - 1) // 2
        pad = (kernel_scale - 1) // 2

        min1, max1, min2, max2 = dim_resizer
        rank_dim = min(max(dim, min1), max1)
        rank_4xdim = min(max(dim, min2), max2)
        

        self.dwconv = PRConv2dLite(dim, dim, kernel_size=kernel_scale, padding=pad, groups=dim, 
                                    rank_group=(rank_dim, rank_dim, rank_k, rank_k))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = PRConv2dLite(dim, 4 * dim, kernel_size=1, rank_group=(rank_dim, rank_4xdim, 1, 1))
        self.act = nn.GELU()
        self.pwconv2 = PRConv2dLite(4 * dim, dim, kernel_size=1, rank_group=(rank_4xdim, rank_dim, 1, 1))
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)).view(1, dim, 1, 1))
        else:
            self.gamma = None
        
        self.norm_z = LayerNorm_Without_Shape(embed_dim*4, BiasFree_LayerNorm)
        self.ln_z = nn.Linear(embed_dim*4, dim)
        self.ln_z2 = nn.Linear(embed_dim*4, dim*4) # for self.pwconv2

    def cast_prior(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 3, "Input z must have shape (batch_size, N, in_channels)"
        b, N, c_in = z.shape
        z = self.norm_z(z)
        z = self.ln_z(z)
        return z

    def cast_prior_2(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 3, "Input z must have shape (batch_size, N, in_channels)"
        b, N, c_in = z.shape
        z = self.norm_z(z)
        z = self.ln_z2(z)
        return z
    
    def forward(self, x, prior=None):  # x: B x C x H x W
        input = x
        
        # depthwise
        x = self.dwconv(x, self.cast_prior(prior))
        x = x.permute(0, 2, 3, 1)  # B H W C


        # norm
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B C H W


        # FFN
        x = self.pwconv1(x, self.cast_prior(prior))
        x = self.act(x)
        x = self.pwconv2(x, self.cast_prior_2(prior))


        if self.gamma is not None:
            x = self.gamma * x
        return input + x


class TensorDecompNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_dim, rank1, rank2, rank3, rank4, kernel_num=1, base_weight=None, groups=1, bias=False):
        super(TensorDecompNetwork, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.feature_dim = feature_dim
        self.rank1, self.rank2, self.rank3, self.rank4 = rank1, rank2, rank3, rank4
        self.kernel_num = kernel_num
        self.weight = base_weight
        self.groups = groups
        self.has_bias = bias

        # Network to generate multi-kernel G
        self.p_to_G = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),  
            nn.ReLU(),
            nn.Linear(in_channels * 2, kernel_num * rank1 * rank2 * rank3 * rank4),
            nn.ReLU(),
            nn.Unflatten(1, (kernel_num, rank1, rank2, rank3, rank4))
        )

        # Kernel attention network
        self.kernel_attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, kernel_num),
            nn.Softmax(dim=1)
        )

        # Feature processor
        self.x_processor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.ReLU()
        )

        self.x_bias_processer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  
            nn.ReLU()
        )

        # Factor matrix generators
        self.x_to_A = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, rank1 * in_channels),
            nn.Unflatten(1, (rank1, in_channels)),
            nn.Sigmoid()
        )
        self.x_to_B = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, rank2 * out_channels),
            nn.Unflatten(1, (rank2, out_channels)),
            nn.Sigmoid()
        )
        self.x_to_C = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, rank3 * kernel_size),
            nn.Unflatten(1, (rank3, kernel_size)),
            nn.Sigmoid()
        )
        self.x_to_D = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, rank4 * kernel_size),
            nn.Unflatten(1, (rank4, kernel_size)),
            nn.Sigmoid()
        )

        # Multi-kernel bias generator (conditional)
        if self.has_bias:
            self.x_to_bias = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, kernel_num * out_channels),
                nn.Unflatten(1, (kernel_num, out_channels))
            )

    def forward(self, x, p):
        batch_size = x.shape[0]

        # Generate multi-kernel components
        G = self.p_to_G(p.mean(dim=1))
        x_feat = self.x_processor(x)
        x_bias_feat = self.x_bias_processer(x)

        # Factor matrices
        A = self.x_to_A(x_feat)
        B = self.x_to_B(x_feat)
        C = self.x_to_C(x_feat)
        D = self.x_to_D(x_feat)

        # Kernel attention
        kernel_attn = self.kernel_attention_net(x_feat)

        # Tensor decomposition
        W = torch.einsum('bkimnp, biq -> bkmnpq', G, A)
        W = torch.einsum('bkmnpq, bmr -> bknpqr', W, B)
        W = torch.einsum('bknpqr, bns -> bkpqrs', W, C)
        W = torch.einsum('bkpqrs, bpt -> bkqrst', W, D)

        # Weight fusion
        b, n, ci, co, k, _ = W.shape
        W = torch.sum(W.view(b, n, -1, ci // self.groups, co, k, k), dim=2)
        W = W * self.weight.permute(0, 2, 1, 3, 4).unsqueeze(0)
        W = torch.einsum('bkqrst, bk -> bqrst', W, kernel_attn)

        # Bias generation (conditional)
        bias = None
        if self.has_bias:
            multi_bias = self.x_to_bias(x_bias_feat)  # (b, kernel_num, out_channels)
            bias = torch.einsum('bk, bkc -> bc', kernel_attn, multi_bias)

        return W, bias


class PRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, reduction=0.0625, kernel_num=1, rank_group=(8, 8, 1, 1)):
        super(PRConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.has_bias = bias

        # Base weight parameter
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, *((kernel_size,) * 2)),
            requires_grad=True
        )
        self._initialize_weights()

        # Decomposition network
        rank1, rank2, rank3, rank4 = rank_group
        self.decomp = TensorDecompNetwork(
            in_channels, out_channels, kernel_size, feature_dim=32,
            rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4, kernel_num=kernel_num,
            base_weight=self.weight, groups=groups, bias=bias
        )

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x, p):
        # Dynamic weight and bias generation
        weight, bias = self.decomp(x, p)

        # Reshape for grouped convolution
        b, c_in, c_out, k, _ = weight.shape
        weight = weight.view(-1, c_in, k, k)

        # Perform convolution
        x = x.reshape(1, -1, *x.shape[2:])
        output = F.conv2d(
            x,
            weight=weight,
            bias=None,  # Apply bias manually later
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=b * self.groups
        )

        # Reshape
        output = output.view(b, -1, *output.shape[2:])

        # Add bias if enabled
        if self.has_bias and bias is not None:
            output += bias.unsqueeze(-1).unsqueeze(-1)  # Add channel-wise bias

        return output


class TensorDecompNetworkLite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_dim, rank1, rank2, rank3, rank4, 
                 kernel_num=1, base_weight=None, groups=1, bias=False, g_factor=1):
        super(TensorDecompNetworkLite, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank1, self.rank2, self.rank3, self.rank4 = rank1, rank2, rank3, rank4
        self.kernel_num = kernel_num
        self.weight = base_weight
        self.groups = groups
        self.has_bias = bias

        # Lightweight G generator
        self.p_to_G = nn.Sequential(
            nn.Linear(in_channels, in_channels*g_factor),
            nn.ReLU(),
            nn.Linear(in_channels*g_factor, kernel_num * rank1 * rank2 * rank3 * rank4),
            nn.Unflatten(1, (kernel_num, rank1, rank2, rank3, rank4)))
        
        # Shared feature processor
        self.feature_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv2d(in_channels, 16, 1),  # Pointwise
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 16, 1),
            nn.ReLU())
        
        # Unified factor generator
        total_factors = (rank1*in_channels + rank2*out_channels + 
                        rank3*kernel_size + rank4*kernel_size)
        self.factor_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, total_factors))
        self.split_sizes = [rank1*in_channels, rank2*out_channels,
                           rank3*kernel_size, rank4*kernel_size]
        
        # Lightweight attention
        self.kernel_attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, kernel_num),
            nn.Softmax(dim=1))
        
        # Bias generator
        if self.has_bias:
            self.x_to_bias = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(16, kernel_num * out_channels),
                nn.Unflatten(1, (kernel_num, out_channels)))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, p):
        batch_size = x.shape[0]
        
        # Feature extraction
        x_feat = self.feature_processor(x)
        # print("x_feat: ", x_feat.shape)
        
        # Generate core tensor G
        G = self.p_to_G(p.mean(dim=1))  # [B, K, R1, R2, R3, R4]
        
        # Generate factor matrices
        factors = self.factor_generator(x_feat)
        A, B, C, D = torch.split(factors, self.split_sizes, dim=1)
        A = self.sigmoid(A.view(batch_size, self.rank1, self.in_channels))  # [B, R1, Cin]
        B = self.sigmoid(B.view(batch_size, self.rank2, self.out_channels))  # [B, R2, Cout]
        C = self.sigmoid(C.view(batch_size, self.rank3, self.kernel_size))   # [B, R3, K]
        D = self.sigmoid(D.view(batch_size, self.rank4, self.kernel_size))   # [B, R4, K]
        
        # Tensor contraction
        W = torch.einsum('bkimnp, biq -> bkmnpq', G, A)
        W = torch.einsum('bkmnpq, bmr -> bknpqr', W, B)
        W = torch.einsum('bknpqr, bns -> bkpqrs', W, C)
        W = torch.einsum('bkpqrs, bpt -> bkqrst', W, D)
        

        # Combine with base weights
        b, n, ci, co, k, _ = W.shape
        W = torch.sum(W.view(b, n, -1, ci // self.groups, co, k, k), dim=2)
        W = W * self.weight.permute(0,2,1,3,4).unsqueeze(0)  # [B, K, Cout, Cin/G, K, K]
        
        
        # Attention fusion
        attn = self.kernel_attention_net(x_feat)  # [B, K]
        fused_W = torch.einsum('bk... ,bk->b...', W, attn)  # [B, Cout, Cin/G, K, K]
        
        # Bias generation
        bias = None
        if self.has_bias:
            multi_bias = self.x_to_bias(x_feat)  # [B, K, Cout]
            bias = torch.einsum('bk,bkc->bc', attn, multi_bias)
        
        return fused_W, bias

class PRConv2dLite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, kernel_num=1, rank_group=(4,4,1,1), g_factor=1):
        super(PRConv2dLite, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.has_bias = bias

        # Base weight tensor
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels//groups, kernel_size, kernel_size),
            requires_grad=True)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        # Decomposition network
        self.decomp = TensorDecompNetworkLite(
            in_channels, out_channels, kernel_size, feature_dim=16,
            rank1=rank_group[0], rank2=rank_group[1], 
            rank3=rank_group[2], rank4=rank_group[3],
            kernel_num=kernel_num, base_weight=self.weight,
            groups=groups, bias=bias, g_factor=g_factor)

    def forward(self, x, p):
        # Dynamic weight and bias generation
        weight, bias = self.decomp(x, p)

        # Reshape for grouped convolution
        b, c_in, c_out, k, _ = weight.shape
        weight = weight.view(-1, c_in, k, k)

        # Perform convolution
        x = x.reshape(1, -1, *x.shape[2:])
        output = F.conv2d(
            x,
            weight=weight,
            bias=None,  # Apply bias manually later
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=b * self.groups
        )

        # Reshape
        output = output.view(b, -1, *output.shape[2:])

        # Add bias if enabled
        if self.has_bias and bias is not None:
            output += bias.unsqueeze(-1).unsqueeze(-1)  # Add channel-wise bias

        return output

class TensorDecompTransposeNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_dim, rank1, rank2, rank3, rank4, kernel_num=1, base_weight=None, groups=1, bias=False):
        super(TensorDecompTransposeNetwork, self).__init__()
        # Structure similar to original TensorDecompNetwork, but input/output channel dimensions are swapped
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.feature_dim = feature_dim
        self.rank1, self.rank2, self.rank3, self.rank4 = rank1, rank2, rank3, rank4
        self.kernel_num = kernel_num
        self.weight = base_weight  # Note: Transpose convolution weight shape differs
        self.groups = groups
        self.has_bias = bias

        # Network structure remains similar, but dimensions may need adjustment
        self.p_to_G = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, kernel_num * rank1 * rank2 * rank3 * rank4),
            nn.ReLU(),
            nn.Unflatten(1, (kernel_num, rank1, rank2, rank3, rank4))
        )

        # Attention network remains unchanged
        self.kernel_attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, kernel_num),
            nn.Softmax(dim=1)
        )

        # Feature processor remains unchanged
        self.x_processor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.x_bias_processor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Factor matrices swap input/output channel dimensions
        self.x_to_A = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, rank1 * out_channels),  # Original out_channels -> in_channels
            nn.Unflatten(1, (rank1, out_channels)),
            nn.Sigmoid()
        )

        self.x_to_B = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, rank2 * in_channels),  # Original in_channels -> out_channels
            nn.Unflatten(1, (rank2, in_channels)),
            nn.Sigmoid()
        )

        self.x_to_C = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, rank3 * kernel_size),
            nn.Unflatten(1, (rank3, kernel_size)),
            nn.Sigmoid()
        )

        self.x_to_D = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, rank4 * kernel_size),
            nn.Unflatten(1, (rank4, kernel_size)),
            nn.Sigmoid()
        )

        # Bias generator maintains out_channels dimension
        if self.has_bias:
            self.x_to_bias = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, kernel_num * out_channels),
                nn.Unflatten(1, (kernel_num, out_channels))
            )

    def forward(self, x, p):
        batch_size = x.shape[0]
        G = self.p_to_G(p.mean(dim=1))
        x_feat = self.x_processor(x)
        x_bias_feat = self.x_bias_processor(x)

        # Swap channel dimensions for A and B
        A = self.x_to_A(x_feat)  # (b, rank1, out_channels)
        B = self.x_to_B(x_feat)  # (b, rank2, in_channels)
        C = self.x_to_C(x_feat)
        D = self.x_to_D(x_feat)

        kernel_attn = self.kernel_attention_net(x_feat)

        # Adjust tensor contraction order to match transpose convolution dimensions
        W = torch.einsum('bkimnp, biq -> bkmnpq', G, A)
        W = torch.einsum('bkmnpq, bmr -> bknpqr', W, B)
        W = torch.einsum('bknpqr, bns -> bkpqrs', W, C)
        W = torch.einsum('bkpqrs, bpt -> bkqrst', W, D)

        # Adjust weight shape to match transpose convolution (in_channels, out_channels/groups, ...)
        b, n, ci, co, k, _ = W.shape
        W = torch.sum(W.view(b, n, -1, co // self.groups, ci, k, k), dim=2)  # Swap ci/co
        # print("W: ",W.shape)
        # print("W0: ", self.weight.unsqueeze(0).shape)
        W = W * self.weight.unsqueeze(0)  # Adjust dimension order
        W = torch.einsum('bkqrst, bk -> bqrst', W, kernel_attn)
        #print("W: ", W.shape)

        # Bias maintains out_channels
        bias = None
        if self.has_bias:
            multi_bias = self.x_to_bias(x_bias_feat)
            bias = torch.einsum('bk, bkc -> bc', kernel_attn, multi_bias)

        return W, bias


class PRConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=False,
                 reduction=0.0625, kernel_num=1, rank_group=(8, 8, 1, 1)):
        super(PRConvTranspose2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding  # Transpose convolution specific parameter
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.has_bias = bias

        # Adjust base weight shape for transpose convolution (in_channels, out_channels//groups, k, k)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, in_channels, out_channels // groups, *((kernel_size,) * 2)),
            requires_grad=True
        )
        self._initialize_weights()

        # Use transpose decomposition network
        rank1, rank2, rank3, rank4 = rank_group
        self.decomp = TensorDecompTransposeNetwork(
            in_channels, out_channels, kernel_size, feature_dim=32,
            rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4, kernel_num=kernel_num,
            base_weight=self.weight, groups=groups, bias=bias
        )

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_in', nonlinearity='relu')  # Transpose convolution uses fan_in

    def forward(self, x, p):
        weight, bias = self.decomp(x, p)  # (b, in_channels, out_channels//groups, k, k)

        # Adjust weight shape for transpose convolution
        b, c_in, c_out_per_group, k, _ = weight.shape
        weight = weight.view(-1, c_out_per_group, k, k)

        # Use conv_transpose2d
        x = x.reshape(1, -1, *x.shape[2:])
        output = F.conv_transpose2d(
            input=x,
            weight=weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,  # New parameter
            dilation=self.dilation,
            groups=b * self.groups
        )

        output = output.view(b, -1, *output.shape[2:])

        if self.has_bias and bias is not None:
            output += bias.unsqueeze(-1).unsqueeze(-1)

        return output


class TensorDecompTransposeNetworkLite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_dim, 
    rank1, rank2, rank3, rank4, kernel_num=1, base_weight=None, groups=1, bias=False, g_factor=1):
        super(TensorDecompTransposeNetworkLite, self).__init__()
        # Structure similar to original TensorDecompNetwork, but input/output channel dimensions are swapped
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.feature_dim = feature_dim
        self.rank1, self.rank2, self.rank3, self.rank4 = rank1, rank2, rank3, rank4
        self.kernel_num = kernel_num
        self.weight = base_weight  # Note: Transpose convolution weight shape differs
        self.groups = groups
        self.has_bias = bias

        # Lightweight G generator
        self.p_to_G = nn.Sequential(
            nn.Linear(in_channels, in_channels*g_factor),
            nn.ReLU(),
            nn.Linear(in_channels*g_factor, kernel_num * rank1 * rank2 * rank3 * rank4),
            nn.Unflatten(1, (kernel_num, rank1, rank2, rank3, rank4)))
        
        # Shared feature processor
        self.feature_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv2d(in_channels, 16, 1),  # Pointwise
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 16, 1),
            nn.ReLU())
        
        # Unified factor generator
        total_factors = (rank1*in_channels + rank2*out_channels + 
                        rank3*kernel_size + rank4*kernel_size)
        self.factor_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, total_factors))
        self.split_sizes = [rank1*in_channels, rank2*out_channels,
                           rank3*kernel_size, rank4*kernel_size]
        
        # Lightweight attention
        self.kernel_attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, kernel_num),
            nn.Softmax(dim=1))
        
        # Bias generator
        if self.has_bias:
            self.x_to_bias = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(16, kernel_num * out_channels),
                nn.Unflatten(1, (kernel_num, out_channels)))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, p):
        batch_size = x.shape[0]
        
        # Feature extraction
        x_feat = self.feature_processor(x)
        
        # Generate core tensor G
        G = self.p_to_G(p.mean(dim=1))  # [B, K, R1, R2, R3, R4]
        
        # Generate factor matrices
        factors = self.factor_generator(x_feat)
        B, A, C, D = torch.split(factors, self.split_sizes, dim=1)
        A = self.sigmoid(A.view(batch_size, self.rank2, self.out_channels))  # [B, R2, Cout]
        B = self.sigmoid(B.view(batch_size, self.rank1, self.in_channels))  # [B, R1, Cin]
        C = self.sigmoid(C.view(batch_size, self.rank3, self.kernel_size))   # [B, R3, K]
        D = self.sigmoid(D.view(batch_size, self.rank4, self.kernel_size))   # [B, R4, K]
        
        # Adjust tensor contraction order to match transpose convolution dimensions
        W = torch.einsum('bkimnp, biq -> bkmnpq', G, A)
        W = torch.einsum('bkmnpq, bmr -> bknpqr', W, B)
        W = torch.einsum('bknpqr, bns -> bkpqrs', W, C)
        W = torch.einsum('bkpqrs, bpt -> bkqrst', W, D)
        

         # Adjust weight shape to match transpose convolution (in_channels, out_channels/groups, ...)
        b, n, ci, co, k, _ = W.shape
        W = torch.sum(W.view(b, n, -1, co // self.groups, ci, k, k), dim=2)  # Swap ci/co
        # print("W: ",W.shape)
        # print("W0: ", self.weight.unsqueeze(0).shape)
        W = W * self.weight.unsqueeze(0)  # Adjust dimension order
        
        #print("W: ", W.shape)
        
        
        # Attention fusion
        attn = self.kernel_attention_net(x_feat)  # [B, K]
        fused_W = torch.einsum('bk... ,bk->b...', W, attn)  # [B, Cout, Cin/G, K, K]
        
        # Bias generation
        bias = None
        if self.has_bias:
            multi_bias = self.x_to_bias(x_feat)  # [B, K, Cout]
            bias = torch.einsum('bk,bkc->bc', attn, multi_bias)
        
        return fused_W, bias

class PRConvTranspose2dLite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=False,
                 reduction=0.0625, kernel_num=1, rank_group=(8, 8, 1, 1), g_factor=1):
        super(PRConvTranspose2dLite, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding  # Transpose convolution specific parameter
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.has_bias = bias

        # Adjust base weight shape for transpose convolution (in_channels, out_channels//groups, k, k)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, in_channels, out_channels // groups, *((kernel_size,) * 2)),
            requires_grad=True
        )
        self._initialize_weights()

        # Use transpose decomposition network
        rank1, rank2, rank3, rank4 = rank_group
        self.decomp = TensorDecompTransposeNetworkLite(
            in_channels, out_channels, kernel_size, feature_dim=32,
            rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4, kernel_num=kernel_num,
            base_weight=self.weight, groups=groups, bias=bias, g_factor=g_factor
        )

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_in', nonlinearity='relu')  # Transpose convolution uses fan_in

    def forward(self, x, p):
        weight, bias = self.decomp(x, p)  # (b, in_channels, out_channels//groups, k, k)

        # Adjust weight shape for transpose convolution
        b, c_in, c_out_per_group, k, _ = weight.shape
        weight = weight.view(-1, c_out_per_group, k, k)

        # Use conv_transpose2d
        x = x.reshape(1, -1, *x.shape[2:])
        output = F.conv_transpose2d(
            input=x,
            weight=weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,  # New parameter
            dilation=self.dilation,
            groups=b * self.groups
        )

        output = output.view(b, -1, *output.shape[2:])

        if self.has_bias and bias is not None:
            output += bias.unsqueeze(-1).unsqueeze(-1)

        return output




if __name__ == "__main__":
    x = torch.randn(8, 32, 256, 256) # b Cin H W
    p = torch.randn(8, 16, 32) # N Cin
    
    conv2 = PRConv2dLite(
                    in_channels=32,       
                    out_channels=32,     
                    kernel_size=7, padding=3, groups=1, rank_group=(32//4, 32//4, 3, 3),
    )
    
    output = conv2(x, p)


    # x = torch.randn(8, 32, 64, 64) 
    # p = torch.randn(8, 16, 32) 

    # conv_trans = PRConvTranspose2d(
    #     in_channels=32,
    #     out_channels=64,  
    #     kernel_size=2,
    #     stride=2,         
    #     padding=0,
    #     output_padding=0, 
    #     groups=1,
    #     kernel_num=1,
    #     rank_group=(8,8,2,2)
    # )

    # output = conv_trans(x, p)  # 输出形状应为 (8, 64, 128, 128)


    print("output:", output.shape)