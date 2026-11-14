import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastcore.all as fc
import warnings

class DilatedDSConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(self.bn(x))

        return x

class ECA(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size=3
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sm = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        feat = self.avg_pool(x)
        feat = feat.squeeze(-1).transpose(-1, -2)
        feat = self.conv(feat)
        feat = feat.transpose(-1, -2).unsqueeze(-1)
        feat = self.sm(feat)

        return x * feat.expand_as(x)

class TemporalGate(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True),
                       x.amax(1, keepdim=True)], dim=1)

        return x * F.sigmoid(self.conv(s))

class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=64,
            R: int=2,
            kernel_size: int=1,
            dilation: list|np.ndarray=[],
            eca_kernel_size: int=3,
            tempo_gate_kernel_size: int=7
    ):
        super().__init__()

        if len(dilation) > R:
            warnings.warn(f"The length of the dilation size list is currently greater than the repeated times ({dilation}, while R={R})")
        if len(dilation) < R:
            raise ValueError(f"Dilation list length does not match repeated time R (expected {R}, got {len(dilation)})")

        self.dds = nn.Sequential()
        c = in_channels
        for i in range(R):
            self.dds.append(DilatedDSConv(
                                c,
                                out_channels,
                                kernel_size,
                                dilation[i]
                            ))
            c = out_channels

        self.eca = ECA(channels=out_channels, kernel_size=eca_kernel_size)
        self.gate = TemporalGate(k=tempo_gate_kernel_size)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor):
        feat = self.dds(x)
        feat = self.eca(feat)
        b,c,h,w = feat.shape
        feat = self.gate(feat.reshape(b,c,-1))
        feat = feat.reshape(b,c,h,w) + self.shortcut(x)

        return feat

@fc.delegates(BasicBlock)
class Student(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            B: int=3,
            Rs: list|np.ndarray=[2, 2, 3],
            Ks: list|np.ndarray=[1, 3, 5],
            Ds: list|np.ndarray=[[1, 3], [1, 3], [1, 3, 5]],
            num_classes: int=1000,
            **kwargs
    ):
        super().__init__()

        self.blocks = nn.Sequential()
        for i in range(B):
            self.blocks.append(BasicBlock(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    R=Rs[i],
                                    kernel_size=Ks[i],
                                    dilation=Ds[i],
                                    **kwargs
                              ))
            in_channels = out_channels

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(out_channels, 512), nn.ReLU(), nn.Linear(512, num_classes))

    def forward(self, x: torch.Tensor):
        if x.dim() == 3: x = x.unsqueeze(1)
        
        feat = self.blocks(x)
        feat = self.avg_pooling(feat).flatten(1)
        logits = self.fc(feat)

        return logits

class Student(nn.Module):
    def __init__(self, d_args):
        super().__init__()

        self.d_args = d_args
        in_channels = 1
        out_channels = d_args['out_channels']
        B = d_args['B']
        Rs = d_args['Rs']
        Ks = d_args['Ks']
        Ds = d_args['Ds']
        eca_kernel_size = d_args['eca_kernel_size']
        tempo_gate_kernel_size = d_args['tempo_gate_kernel_size']

        self.blocks = nn.Sequential()
        for i in range(B):
            self.blocks.append(BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                R=Rs[i],
                kernel_size=Ks[i],
                dilation=Ds[i],
                eca_kernel_size=eca_kernel_size,
                tempo_gate_kernel_size=tempo_gate_kernel_size
            ))
            in_channels = out_channels

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(out_channels, 512), nn.ReLU(), nn.Linear(512, 2))

    def forward(self, x: torch.Tensor):
        if x.dim() == 3: x = x.unsqueeze(1)
        # Encoder
        feat = self.blocks(x) # (B, C, M, N)
        feat = self.avg_pooling(feat).flatten(1) # (B, CxMxN)
        logits = self.fc(feat) # (B, 2)

        return logits