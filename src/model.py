import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, d=1, groups=1):
        super().__init__()
        p = d * (k - 1) // 2
        self.conv = nn.Conv2d(cin, cout, k, s, p, dilation=d, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWSepResBlock(nn.Module):

    def __init__(self, ch, d=1):
        super().__init__()
        self.dw = ConvBNAct(ch, ch, k=3, d=d, groups=ch)
        self.pw = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.pw(self.dw(x)))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, s=1):
        super().__init__()
        self.c1 = ConvBNAct(cin, cout, s=s)
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, s, bias=False), nn.BatchNorm2d(cout))
            if (cin != cout or s != 1) else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.skip(x) + self.c2(self.c1(x)))


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r), nn.SiLU(inplace=True), nn.Linear(ch // r, ch), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))
        return x * w[:, :, None, None]


class SpectralStem(nn.Module):

    def __init__(self, out_ch=32):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(6, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True),
        )
        self.attn = ChannelAttention(out_ch)
        self.res = nn.Sequential(DWSepResBlock(out_ch), DWSepResBlock(out_ch))

    @staticmethod
    def aux_channels(x):
        g, r, nir = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        ndvi = (nir - r) / (nir + r + EPS)
        igr = (g - r) / (g + r + EPS)
        imean = (g + r + nir) / 3.0
        return torch.cat([g, r, nir, ndvi, igr, imean], dim=1)

    def forward(self, x):
        f = self.mix(self.aux_channels(x))
        return self.res(self.attn(f))


class BiFusion(nn.Module):

    def __init__(self, chs):
        super().__init__()
        c4, c8, c16 = chs
        self.p8to4 = nn.Conv2d(c8, c4, 1, bias=False)
        self.p16to8 = nn.Conv2d(c16, c8, 1, bias=False)
        self.p4to8 = nn.Conv2d(c4, c8, 3, 2, 1, bias=False)
        self.p8to16 = nn.Conv2d(c8, c16, 3, 2, 1, bias=False)
        self.r4 = DWSepResBlock(c4)
        self.r8 = DWSepResBlock(c8)
        self.r16 = DWSepResBlock(c16)

    def forward(self, f4, f8, f16):
        up8 = F.interpolate(self.p8to4(f8), size=f4.shape[-2:], mode="bilinear", align_corners=False)
        up16 = F.interpolate(self.p16to8(f16), size=f8.shape[-2:], mode="bilinear", align_corners=False)
        o4 = self.r4(f4 + up8)
        o8 = self.r8(f8 + up16 + self.p4to8(f4))
        o16 = self.r16(f16 + self.p8to16(f8))
        return o4, o8, o16


class DualPathEncoder(nn.Module):
    def __init__(self, in_ch=32, c4=64, c8=128, c16=256, n_fusion=2):
        super().__init__()
        self.down4 = nn.Sequential(ResBlock(in_ch, c4 // 2, s=2), ResBlock(c4 // 2, c4, s=2))
        self.hr = nn.Sequential(
            DWSepResBlock(c4, d=1), DWSepResBlock(c4, d=2), DWSepResBlock(c4, d=3),
            DWSepResBlock(c4, d=1), DWSepResBlock(c4, d=2), DWSepResBlock(c4, d=3),
        )
        self.down8 = nn.Sequential(ResBlock(c4, c8, s=2), ResBlock(c8, c8))
        self.down16 = nn.Sequential(ResBlock(c8, c16, s=2), ResBlock(c16, c16))
        self.fusions = nn.ModuleList([BiFusion((c4, c8, c16)) for _ in range(n_fusion)])

    def forward(self, x):
        f4 = self.down4(x)
        f4 = self.hr(f4)
        f8 = self.down8(f4)
        f16 = self.down16(f8)
        for fuse in self.fusions:
            f4, f8, f16 = fuse(f4, f8, f16)
        return f4, f8, f16


class TopologyDecoder(nn.Module):

    def __init__(self, stem_ch=32, c4=64, c8=128, c16=256, dec_ch=48, hr_refine=False, learned_down2=False,
                 sep_heads=False):
        super().__init__()
        self.hr_refine, self.learned_down2, self.sep_heads = hr_refine, learned_down2, sep_heads
        if hr_refine:
            self.refine1 = ResBlock(dec_ch, dec_ch)
        if learned_down2:
            self.down2 = nn.Sequential(ConvBNAct(stem_ch, stem_ch, s=2), DWSepResBlock(stem_ch))
        if sep_heads:
            self.branch_mask = ConvBNAct(dec_ch, dec_ch)
            self.branch_topo = ConvBNAct(dec_ch, dec_ch)
        self.lat8 = nn.Conv2d(c8, c4, 1, bias=False)
        self.lat16 = nn.Conv2d(c16, c4, 1, bias=False)
        self.merge4 = ResBlock(c4, c4)
        self.up2 = ResBlock(c4 + stem_ch, dec_ch)
        self.up1 = ResBlock(dec_ch + stem_ch, dec_ch)
        self.head_mask = nn.Conv2d(dec_ch, 1, 1)
        self.head_skel = nn.Conv2d(dec_ch, 1, 1)
        self.head_width = nn.Conv2d(dec_ch, 1, 1)
        self.head_aff = nn.Conv2d(dec_ch, 8, 1)
        self.head_aux = nn.Conv2d(c4, 1, 1)

    def forward(self, stem, f4, f8, f16):
        h, w = f4.shape[-2:]
        f = f4 \
            + F.interpolate(self.lat8(f8), size=(h, w), mode="bilinear", align_corners=False) \
            + F.interpolate(self.lat16(f16), size=(h, w), mode="bilinear", align_corners=False)
        f = self.merge4(f)
        aux_logit = self.head_aux(f)
        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        stem_half = self.down2(stem) if self.learned_down2 else F.avg_pool2d(stem, 2)
        f = self.up2(torch.cat([f, stem_half], dim=1))
        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.up1(torch.cat([f, stem], dim=1))
        if self.hr_refine:
            f = self.refine1(f)
        fm = self.branch_mask(f) if self.sep_heads else f
        ft = self.branch_topo(f) if self.sep_heads else f
        return {
            "mask_logit": self.head_mask(fm),
            "skel_logit": self.head_skel(ft),
            "width": F.softplus(self.head_width(ft)) + 0.5,
            "aff_logit": self.head_aff(ft),
            "aux_logit": aux_logit,
        }


def reconstruct_mask(p_skel, width, k=9):
    b, _, h, w = p_skel.shape
    pad = k // 2
    ps_u = F.unfold(p_skel, k, padding=pad)
    w_u = F.unfold(width, k, padding=pad)
    dy, dx = torch.meshgrid(
        torch.arange(-pad, pad + 1, device=p_skel.device, dtype=p_skel.dtype),
        torch.arange(-pad, pad + 1, device=p_skel.device, dtype=p_skel.dtype),
        indexing="ij",
    )
    d2 = (dy ** 2 + dx ** 2).reshape(1, -1, 1)
    contrib = ps_u * torch.exp(-d2 / (2.0 * w_u ** 2 + EPS))
    return contrib.max(dim=1).values.view(b, 1, h, w)


class LISSTopoRoadNet(nn.Module):

    def __init__(self, stem_ch=32, c4=64, c8=128, c16=256, dec_ch=48, recon_k=9,
                 use_spectral_stem=True, use_topology_heads=True, use_reconstruction=True,
                 drop_topology_heads=False, hr_refine=False, learned_down2=False, sep_heads=False):
        super().__init__()
        self.use_topology_heads = use_topology_heads
        self.drop_topology_heads = drop_topology_heads
        self.use_reconstruction = use_reconstruction and use_topology_heads and not drop_topology_heads
        self.recon_k = recon_k
        if use_spectral_stem:
            self.stem = SpectralStem(stem_ch)
        else:
            self.stem = nn.Sequential(
                ConvBNAct(3, stem_ch), DWSepResBlock(stem_ch), DWSepResBlock(stem_ch)
            )
        self.encoder = DualPathEncoder(stem_ch, c4, c8, c16)
        self.decoder = TopologyDecoder(stem_ch, c4, c8, c16, dec_ch, hr_refine=hr_refine,
                                       learned_down2=learned_down2, sep_heads=sep_heads)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        stem = self.stem(x)
        f4, f8, f16 = self.encoder(stem)
        out = self.decoder(stem, f4, f8, f16)
        if self.use_reconstruction:
            p_skel = torch.sigmoid(out["skel_logit"])
            m_rec = reconstruct_mask(p_skel, out["width"], k=self.recon_k)
            out["m_rec"] = m_rec
            rec_logit = torch.logit(m_rec.clamp(1e-4, 1 - 1e-4))
            out["fused_logit"] = self.alpha * out["mask_logit"] + self.beta * rec_logit
        else:
            out["fused_logit"] = out["mask_logit"]
        if self.drop_topology_heads:
            for k in ("skel_logit", "width", "aff_logit"):
                out.pop(k, None)
        return out


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
