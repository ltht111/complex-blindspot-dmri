import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexConv2d


class ComplexReflectPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) == 2:
                self.padding = (padding[1], padding[1], padding[0], padding[0])
            elif len(padding) == 4:
                self.padding = padding
            else:
                raise ValueError(f"Unsupported padding tuple length: {len(padding)}")
        else:
            raise TypeError("padding must be int or tuple")

    def forward(self, x):
        if sum(self.padding) == 0:
            return x
        real = F.pad(x.real, self.padding, mode="reflect")
        imag = F.pad(x.imag, self.padding, mode="reflect")
        return torch.complex(real, imag)


class ComplexConv2dReflect(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.pad = ComplexReflectPad2d(padding) if padding else None
        self.conv = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        return self.conv(x)


class ComplexBlur2d(nn.Module):
    def __init__(self, channels, filt_size=3):
        super().__init__()
        if filt_size != 3:
            raise ValueError("Only filt_size=3 is implemented.")
        base = torch.tensor([[1.0, 2.0, 1.0],
                             [2.0, 4.0, 2.0],
                             [1.0, 2.0, 1.0]], dtype=torch.float32)
        base = base / base.sum()
        kernel = base[None, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.channels = channels
        self.pad = (1, 1, 1, 1)

    def forward(self, x):
        real = F.pad(x.real, self.pad, mode="reflect")
        imag = F.pad(x.imag, self.pad, mode="reflect")
        real = F.conv2d(real, self.kernel, stride=1, padding=0, groups=self.channels)
        imag = F.conv2d(imag, self.kernel, stride=1, padding=0, groups=self.channels)
        return torch.complex(real, imag)


class ComplexLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels * 2))
        self.bias = nn.Parameter(torch.zeros(channels * 2))

    def forward(self, x):
        b, c, h, w = x.shape
        x_cat = torch.cat([x.real, x.imag], dim=1).permute(0, 2, 3, 1)
        x_norm = F.layer_norm(x_cat, (2 * c,), self.weight, self.bias, self.eps)
        x_norm = x_norm.permute(0, 3, 1, 2)
        real, imag = torch.chunk(x_norm, 2, dim=1)
        return torch.complex(real, imag)


class ComplexSimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        return x1 * x2


class ComplexSCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ComplexConv2dReflect(channels, channels, kernel_size=1)

    def forward(self, x):
        attn = self.conv(self.pool(x))
        return x * attn


class ComplexNAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        dw_channel = c * 2
        self.norm1 = ComplexLayerNorm(c)
        self.conv1 = ComplexConv2dReflect(c, dw_channel, kernel_size=1)
        self.conv2 = ComplexConv2dReflect(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.sg = ComplexSimpleGate()
        self.sca = ComplexSCA(c)
        self.conv3 = ComplexConv2dReflect(c, c, kernel_size=1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)

    def forward(self, x):
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.sg(out)
        out = self.sca(out)
        out = self.conv3(out)
        return out * self.beta.type_as(out.real) + x


class ComplexAntiAliasDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blur = ComplexBlur2d(in_channels)
        self.proj = ComplexConv2dReflect(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.blur(x)
        return self.proj(x)


class ComplexUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, blur_after_up=True):
        super().__init__()
        self.blur_after_up = blur_after_up
        self.blur = ComplexBlur2d(in_channels) if blur_after_up else nn.Identity()
        self.conv = ComplexConv2dReflect(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        real_up = F.interpolate(x.real, scale_factor=2, mode='bilinear', align_corners=False)
        imag_up = F.interpolate(x.imag, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.complex(real_up, imag_up)
        out = self.blur(out)
        return self.conv(out)


class GuidedComplexNAFNet(nn.Module):
    """
    2.5D single-target no-b0, no-support version.

    Changes in this anti-alias/reflect version:
    - all padded convolutions use explicit reflect padding
    - stride-2 downsampling is replaced with blur + stride-2 conv
    - upsampling keeps bilinear interpolation, then anti-alias blur + reflect-padded conv
    - output remains direct prediction (no residual output restored here)

    Input:
        target_dwi: [B, M, S, H, W] complex or [B, S, H, W] complex
                    where S = num_slices (recommended 3)
    Output:
        center-slice denoised target:
            [B, M, 1, H, W] or [B, 1, H, W]
    """

    def __init__(self, dwi_channels=3, out_channels=1, num_slices=3, support_k=0, max_targets=1):
        super().__init__()
        assert num_slices % 2 == 1, "num_slices must be odd for center-slice prediction."
        if out_channels != 1:
            raise ValueError(
                f"This single-target network must use out_channels=1, got out_channels={out_channels}."
            )
        self.num_slices = num_slices
        self.center_idx = num_slices // 2
        self.support_k = support_k
        self.max_targets = max_targets

        self.intro = ComplexConv2dReflect(dwi_channels, 32, kernel_size=3, padding=1)

        self.enc1 = nn.Sequential(ComplexNAFBlock(32), ComplexNAFBlock(32))
        self.pool1 = ComplexAntiAliasDownsample(32, 64)

        self.enc2 = nn.Sequential(ComplexNAFBlock(64), ComplexNAFBlock(64))
        self.pool2 = ComplexAntiAliasDownsample(64, 128)

        self.enc3 = nn.Sequential(ComplexNAFBlock(128), ComplexNAFBlock(128))
        self.pool3 = ComplexAntiAliasDownsample(128, 256)

        self.bottleneck = nn.Sequential(
            ComplexNAFBlock(256),
            ComplexNAFBlock(256),
            ComplexNAFBlock(256),
        )

        self.up3 = ComplexUpsample(256, 128, blur_after_up=True)
        self.reduce3 = ComplexConv2dReflect(256, 128, kernel_size=1)
        self.dec3 = nn.Sequential(ComplexNAFBlock(128), ComplexNAFBlock(128))

        self.up2 = ComplexUpsample(128, 64, blur_after_up=True)
        self.reduce2 = ComplexConv2dReflect(128, 64, kernel_size=1)
        self.dec2 = nn.Sequential(ComplexNAFBlock(64), ComplexNAFBlock(64))

        self.up1 = ComplexUpsample(64, 32, blur_after_up=True)
        self.reduce1 = ComplexConv2dReflect(64, 32, kernel_size=1)
        self.dec1 = nn.Sequential(ComplexNAFBlock(32), ComplexNAFBlock(32))

        self.out_channels = out_channels
        self.final_conv = ComplexConv2dReflect(32, self.out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _ensure_target_dim(target_dwi):
        squeeze_m = False
        if target_dwi.dim() == 4:
            target_dwi = target_dwi.unsqueeze(1)
            squeeze_m = True
        return target_dwi, squeeze_m

    @staticmethod
    def _pad_if_needed(x):
        h, w = x.shape[2], x.shape[3]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            real = F.pad(x.real, (0, pad_w, 0, pad_h), mode="reflect")
            imag = F.pad(x.imag, (0, pad_w, 0, pad_h), mode="reflect")
            x = torch.complex(real, imag)
        return x, h, w

    @staticmethod
    def _crop_if_needed(x, h, w):
        return x[:, :, :h, :w]

    def forward(self, target_dwi):
        target_dwi, squeeze_m = self._ensure_target_dim(target_dwi)
        b, m, s, h0, w0 = target_dwi.shape
        assert s == self.num_slices, f"Expected {self.num_slices} slices, got {s}"

        target_flat = target_dwi.reshape(b * m, s, h0, w0)
        target_flat, h, w = self._pad_if_needed(target_flat)

        x = self.intro(target_flat)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        mid = self.bottleneck(p3)

        d3 = self.up3(mid)
        d3 = self.dec3(self.reduce3(torch.cat((d3, e3), dim=1)))

        d2 = self.up2(d3)
        d2 = self.dec2(self.reduce2(torch.cat((d2, e2), dim=1)))

        d1 = self.up1(d2)
        d1 = self.dec1(self.reduce1(torch.cat((d1, e1), dim=1)))

        out = self.final_conv(d1)
        out = self._crop_if_needed(out, h, w)

        if out.shape[1] != self.out_channels:
            raise RuntimeError(
                f"Unexpected output channels: expected {self.out_channels}, got {out.shape[1]}."
            )
        out = out.view(b, m, self.out_channels, out.shape[2], out.shape[3])
        if squeeze_m:
            out = out[:, 0]
        return out
