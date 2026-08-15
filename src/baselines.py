"""Baseline models sharing the trainer interface: forward(x) -> {"fused_logit": ...}.

UNet: plain encoder-decoder (the failing reference point).
DLinkNet: ResNet-style encoder + dilated center block + LinkNet decoder
(faithful re-implementation at reduced width; no ImageNet weights since our
input is G/R/NIR, which is itself part of the paper's argument).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList()
        c_prev = in_ch
        for c in chs:
            self.enc.append(conv_block(c_prev, c))
            c_prev = c
        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, 2))
            self.dec.append(conv_block(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, blk, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            x = blk(torch.cat([x, skip], dim=1))
        return {"fused_logit": self.head(x)}


class _ResLayer(nn.Module):
    def __init__(self, cin, cout, n, stride):
        super().__init__()
        layers = []
        for i in range(n):
            s = stride if i == 0 else 1
            layers.append(_BasicBlock(cin if i == 0 else cout, cout, s))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, s):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1, bias=False),
                                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, s, bias=False), nn.BatchNorm2d(cout))
                     if (cin != cout or s != 1) else nn.Identity())

    def forward(self, x):
        return F.relu(self.skip(x) + self.c2(self.c1(x)))


class _DecoderBlock(nn.Module):
    """LinkNet decoder block."""

    def __init__(self, cin, cout):
        super().__init__()
        m = cin // 4
        self.body = nn.Sequential(
            nn.Conv2d(cin, m, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m, m, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(m), nn.ReLU(inplace=True),
            nn.Conv2d(m, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class DLinkNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c[0], 7, 2, 3, bias=False), nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.l1 = _ResLayer(c[0], c[0], 2, 1)
        self.l2 = _ResLayer(c[0], c[1], 2, 2)
        self.l3 = _ResLayer(c[1], c[2], 2, 2)
        self.l4 = _ResLayer(c[2], c[3], 2, 2)
        # dilated center block (D-LinkNet's key component), parallel-sum of cascades
        self.dil = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c[3], c[3], 3, 1, d, dilation=d, bias=False),
                          nn.BatchNorm2d(c[3]), nn.ReLU(inplace=True))
            for d in (1, 2, 4, 8)])
        self.d4 = _DecoderBlock(c[3], c[2])
        self.d3 = _DecoderBlock(c[2], c[1])
        self.d2 = _DecoderBlock(c[1], c[0])
        self.d1 = _DecoderBlock(c[0], c[0])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(c[0], c[0] // 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c[0] // 2, 1, 3, 1, 1))

    def forward(self, x):
        x = self.stem(x)
        e1 = self.l1(x)
        e2 = self.l2(e1)
        e3 = self.l3(e2)
        e4 = self.l4(e3)
        d = e4
        acc = e4
        for blk in self.dil:
            d = blk(d)
            acc = acc + d
        y = self.d4(acc) + e3
        y = self.d3(y) + e2
        y = self.d2(y) + e1
        y = self.d1(y)
        return {"fused_logit": self.head(y)}


BASELINES = {"unet": UNet, "dlinknet": DLinkNet}


# ------------------------------------------------------------ RoadFormer-lite

class _SRAttention(nn.Module):
    """Efficient self-attention with spatial reduction (SegFormer/PVT style)."""

    def __init__(self, dim, heads, sr):
        super().__init__()
        self.heads, self.sr_ratio = heads, sr
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if sr > 1:
            self.sr = nn.Conv2d(dim, dim, sr, sr)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        if self.sr_ratio > 1:
            xr = x.transpose(1, 2).reshape(b, c, h, w)
            xr = self.sr(xr).reshape(b, c, -1).transpose(1, 2)
            xr = self.norm(xr)
        else:
            xr = x
        kv = self.kv(xr).reshape(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * (c // self.heads) ** -0.5
        out = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class _TFBlock(nn.Module):
    def __init__(self, dim, heads, sr):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = _SRAttention(dim, heads, sr)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, h, w):
        x = x + self.attn(self.n1(x), h, w)
        return x + self.mlp(self.n2(x))


class RoadFormerLite(nn.Module):
    """Compact SegFormer-style hierarchical transformer for road segmentation."""

    def __init__(self, in_ch=3, dims=(32, 64, 128, 192), depths=(2, 2, 2, 2),
                 heads=(1, 2, 4, 8), srs=(8, 4, 2, 1)):
        super().__init__()
        self.stages = nn.ModuleList()
        c_prev = in_ch
        for i, (d, dep, hd, sr) in enumerate(zip(dims, depths, heads, srs)):
            patch = nn.Conv2d(c_prev, d, 7 if i == 0 else 3, 4 if i == 0 else 2,
                              3 if i == 0 else 1)
            blocks = nn.ModuleList([_TFBlock(d, hd, sr) for _ in range(dep)])
            self.stages.append(nn.ModuleList([patch, blocks, nn.LayerNorm(d)]))
            c_prev = d
        self.lin = nn.ModuleList([nn.Conv2d(d, 128, 1) for d in dims])
        self.fuse = nn.Sequential(nn.Conv2d(128 * 4, 128, 1, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = []
        for patch, blocks, norm in self.stages:
            x = patch(x)
            b, c, h, w = x.shape
            t = x.flatten(2).transpose(1, 2)
            for blk in blocks:
                t = blk(t, h, w)
            x = norm(t).transpose(1, 2).reshape(b, c, h, w)
            feats.append(x)
        up = [F.interpolate(lin(f), size=feats[0].shape[-2:], mode="bilinear",
                            align_corners=False) for lin, f in zip(self.lin, feats)]
        y = self.fuse(torch.cat(up, 1))
        y = self.head(y)
        return {"fused_logit": F.interpolate(y, size=(H, W), mode="bilinear",
                                             align_corners=False)}


BASELINES["roadformer"] = RoadFormerLite
