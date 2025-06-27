import torch
from torch import nn
import torch.nn.functional as F
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape)*0.1)
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LargeKernelAttention(nn.Module):
    def __init__(self, channels, kernel_size=21, dilation=3):
        super().__init__()
        d = dilation
        kernel_decomposed = (kernel_size + d - 1) // d

        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=2 * d - 1,
            padding=d - 1,  # (2d-1-1)//2 = d-1
            groups=channels,
            bias=False
        )

        self.dwd_conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_decomposed,
            padding=d * (kernel_decomposed - 1) // 2,  # 关键修正点
            dilation=d,
            groups=channels,
            bias=False
        )

        self.pw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        attention = self.dw_conv(x)
        attention = self.dwd_conv(attention)
        attention = self.pw_conv(attention)
        return x * attention



class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, channels, height, width)


class FFN(nn.Module):
    def __init__(self, in_channels,kernel_size1=5,kernel_size2=3,groups=4):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=kernel_size1,
                      padding=kernel_size1//2, groups=in_channels // 2, bias=False),
        )


        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=kernel_size2,
                      padding=kernel_size2//2, groups=in_channels // 2, bias=False),
        )


        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.LeakyReLU()
        )

        self.shuffle = ChannelShuffle(groups=groups)

    def forward(self, x):

        x1, x2 = torch.split(x, split_size_or_sections=x.size(1) // 2, dim=1)

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        x = torch.cat([x1, x2], dim=1)

        x = self.conv1x1(x)


        return x






class Block(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size=13,
                 dilation=3,
                 kernel_size1=5,
                 kernel_size2=3,
                 groups=4):
        super().__init__()

        self.norm1 = LayerNorm(channels, data_format="channels_first")
        self.lka = LargeKernelAttention(channels, kernel_size, dilation)


        self.norm2 = LayerNorm(channels, data_format="channels_first")
        self.ffn = FFN(in_channels=channels,kernel_size1=kernel_size1,kernel_size2=kernel_size2,groups=groups)

        self.channel_adjust = nn.Identity()

    def forward(self, x):
        identity = x

        x = self.norm1(x)

        x = self.lka(x)

        x = self.norm2(x)
        x = self.ffn(x)

        x = x + identity

        return x


class S2_sample(nn.Module):
    def __init__(self, in_channels, scale=6):
        super().__init__()
        self.scale = scale

        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 2, 1)
        )

        self.scope = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 2, 1, bias=False),
            nn.Sigmoid()
        )
        self.register_buffer('base_grid', self._create_base_grid())


        nn.init.kaiming_normal_(self.offset_net[0].weight, mode='fan_out')
        nn.init.constant_(self.offset_net[-1].weight, 0)
        nn.init.constant_(self.offset_net[-1].bias, 0)
        nn.init.constant_(self.scope[-2].weight, 1 / 16)

    def _create_base_grid(self):

        h = torch.linspace(-1, 1, self.scale)
        grid = torch.stack(torch.meshgrid(h, h), dim=-1)
        return grid.view(1, self.scale, self.scale, 1, 2)

    def forward(self, lr, hr):
        B, C, H, W = lr.shape


        base_offset = self.offset_net(hr)  # [B,2,H,W]
        scope = self.scope(hr)  # [B,2,H,W] ∈ [0,1]
        offset = base_offset * scope * 0.5

        B, _, H, W = offset.shape
        offset = offset.view(B, 2, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(0).type(lr.dtype).to(lr.device).repeat(B, 1, 1, 1)
        normalizer = torch.tensor([W, H], dtype=lr.dtype, device=lr.device).view(1, 2, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.permute(0, 2, 3, 1)

        sampled = F.grid_sample(
            lr, coords,
            mode='bilinear',
        )

        return sampled



class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size=15,
                 dilation=3,
                 kernel_size1=3,
                 kernel_size2=5,
                 groups=4):
        super().__init__()


        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1),
            nn.LeakyReLU(),
        ) if in_channels != channels else nn.Identity()


        self.block = Block(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            kernel_size1=kernel_size1,
            kernel_size2=kernel_size2,
            groups=groups
        )

    def forward(self, x):

        x = self.channel_adjust(x)

        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class S2Mixer(nn.Module):
    def __init__(self,
                 in_channels=12,
                 out_channels=2,
                 base_channels=128,
                 num_blocks=6,
                 LK_size=11,
                 dilation=2,
                 kernel_size1=7,
                 kernel_size2=5,
                 scale_factor=6,
                 increase_channels=64):
        super().__init__()


        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU()
        )


        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        self.num_blocks = num_blocks


        self.dense_blocks.append(
            DenseBlock(
                in_channels=base_channels,
                channels=base_channels,
                kernel_size1=kernel_size1,
                kernel_size2=kernel_size2,
                kernel_size=LK_size,
                dilation=dilation,
                groups=4
            )
        )
        self.trans_layers.append(
            TransitionLayer(
                in_channels=base_channels,
                out_channels=increase_channels
            )
        )


        for i in range(1, num_blocks):
            current_in = base_channels + increase_channels * i
            self.dense_blocks.append(
                DenseBlock(
                    in_channels=current_in,
                    channels=base_channels,
                    kernel_size1=kernel_size1,
                    kernel_size2=kernel_size2,
                    kernel_size=LK_size,
                    dilation=dilation,
                    groups=4
                )
            )
            self.trans_layers.append(
                TransitionLayer(
                    in_channels=base_channels,
                    out_channels=increase_channels
                )
            )


        self.upsampler = S2_sample(base_channels, scale=scale_factor)


        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, input10, input20, input60):

        up60 = F.interpolate(input60, scale_factor=6, mode='bilinear')
        up20 = F.interpolate(input20, scale_factor=2, mode='bilinear')
        x = self.conv_first(torch.cat((input10, up20, up60), dim=1))

        shallow_features = []
        deep_features = x


        deep_out = self.dense_blocks[0](deep_features)
        shallow = self.trans_layers[0](deep_out)
        shallow_features.append(shallow)


        for i in range(1, self.num_blocks):

            combined = torch.cat([deep_out] + shallow_features[:i], dim=1)

            deep_out = self.dense_blocks[i](combined)


            shallow = self.trans_layers[i](deep_out)
            shallow_features.append(shallow)


        deep_out = x + deep_out
        output = self.final_conv(deep_out)


        upsampled = self.upsampler(input60, deep_out)

        return output + upsampled


class S2Mixer_2x(nn.Module):
    def __init__(self,
                 in_channels=10,
                 out_channels=6,
                 base_channels=128,
                 num_blocks=6,
                 LK_size=11,
                 dilation=2,
                 kernel_size1=7,
                 kernel_size2=5,
                 scale_factor=2,
                 increase_channels=64):
        super().__init__()


        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU()
        )


        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        self.num_blocks = num_blocks


        self.dense_blocks.append(
            DenseBlock(
                in_channels=base_channels,
                channels=base_channels,
                kernel_size1=kernel_size1,
                kernel_size2=kernel_size2,
                kernel_size=LK_size,
                dilation=dilation,
                groups=4
            )
        )
        self.trans_layers.append(
            TransitionLayer(
                in_channels=base_channels,
                out_channels=increase_channels
            )
        )


        for i in range(1, num_blocks):
            current_in = base_channels + increase_channels * i
            self.dense_blocks.append(
                DenseBlock(
                    in_channels=current_in,
                    channels=base_channels,
                    kernel_size1=kernel_size1,
                    kernel_size2=kernel_size2,
                    kernel_size=LK_size,
                    dilation=dilation,
                    groups=4
                )
            )
            self.trans_layers.append(
                TransitionLayer(
                    in_channels=base_channels,
                    out_channels=increase_channels
                )
            )

        self.upsampler = S2_sample(base_channels, scale=scale_factor)


        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, input10, input20):

        up20 = F.interpolate(input20, scale_factor=2, mode='bilinear')
        x = self.conv_first(torch.cat((input10, up20), dim=1))


        shallow_features = []
        deep_features = x


        deep_out = self.dense_blocks[0](deep_features)
        shallow = self.trans_layers[0](deep_out)
        shallow_features.append(shallow)

        for i in range(1, self.num_blocks):

            combined = torch.cat([deep_out] + shallow_features[:i], dim=1)


            deep_out = self.dense_blocks[i](combined)


            shallow = self.trans_layers[i](deep_out)
            shallow_features.append(shallow)


        deep_out = x + deep_out
        output = self.final_conv(deep_out)


        upsampled = self.upsampler(input20, deep_out)
        return output + upsampled

