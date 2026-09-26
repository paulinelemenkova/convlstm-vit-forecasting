from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from rasterio.features import rasterize
from torch import nn

IN = Path(os.environ.get("JAXA_DIR", ".")).expanduser()
ADM0 = Path(os.environ.get("VN_ADMIN0", "")).expanduser()
PRED = Path(os.environ.get("PRED_DIR", "./hindcast")).expanduser()
OUT = Path(os.environ.get("FIG_OUT", "./figs")).expanduser()
OUT.mkdir(parents=True, exist_ok=True)


def find_file(name: str, root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"folder does not exist: {root}")
    hits = sorted(p for p in root.rglob(name) if not p.name.startswith("._"))
    if not hits:
        raise FileNotFoundError(f"{name} not found below {root.resolve()}; set JAXA_DIR and HDX_DIR")
    return hits[0]


ADM0 = ADM0 if ADM0.is_file() else find_file("vnm_admin0.shp", Path(os.environ.get("HDX_DIR", ".")).expanduser())

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica"], "font.size": 7.5,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans", "Helvetica"])).lower()

K, T_IN, TILE, P = 7, 5, 64, 8
G = TILE // P
YEARS = list(range(2006, 2011))
GROUPS = {1: [10, 11, 12, 20], 2: [14], 3: [15], 4: [4], 5: [3, 5, 6], 6: [1, 2], 7: [7, 8, 9, 16, 18, 19]}
LUT = np.zeros(256, dtype=np.uint8)
for gid, codes in GROUPS.items():
    LUT[codes] = gid


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.last_att: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, keep: bool) -> torch.Tensor:
        y = self.n1(x)
        a, wts = self.att(y, y, y, need_weights=keep, average_attn_weights=True)
        if keep:
            self.last_att = wts.detach()
        x = x + a
        return x + self.mlp(self.n2(x))


class TemporalViT(nn.Module):
    def __init__(self, p: int = 8, d: int = 128, heads: int = 4, layers: int = 4) -> None:
        super().__init__()
        self.p, self.g = p, TILE // p
        self.embed = nn.Linear(K * p * p, d)
        self.pos = nn.Parameter(torch.zeros(1, 1, self.g * self.g, d))
        self.tpos = nn.Parameter(torch.zeros(1, T_IN, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, K * p * p)

    def forward(self, x: torch.Tensor, keep: bool = False) -> torch.Tensor:
        b, t, k, h, w = x.shape
        p, g = self.p, self.g
        tok = x.reshape(b, t, k, g, p, g, p).permute(0, 1, 3, 5, 2, 4, 6).reshape(b, t, g * g, k * p * p)
        z = (self.embed(tok) + self.pos + self.tpos).reshape(b, t * g * g, -1)
        for blk in self.blocks:
            z = blk(z, keep)
        z = self.norm(z).reshape(b, t, g * g, -1).mean(1)
        out = self.head(z).reshape(b, g, g, K, p, p).permute(0, 3, 1, 4, 2, 5)
        return out.reshape(b, K, h, w)

    def received(self) -> torch.Tensor:
        att0 = self.blocks[0].last_att
        assert att0 is not None
        eye = torch.eye(att0.shape[-1])
        r = eye.expand_as(att0)
        for blk in self.blocks:
            a = 0.5 * blk.last_att + 0.5 * eye
            r = (a / a.sum(-1, keepdim=True)) @ r
        return r.mean(1).reshape(att0.shape[0], T_IN, -1)


with rasterio.open(find_file("VLUCD_L2_250m_2010.tif", IN)) as src:
    tr, crs, shape = src.transform, src.crs, src.shape
adm0 = gpd.read_file(ADM0).to_crs(crs)
inside = rasterize(((g, 1) for g in adm0.geometry), out_shape=shape, transform=tr, dtype="uint8") == 1
stack = np.zeros((T_IN, *shape), dtype=np.uint8)
for i, y in enumerate(YEARS):
    with rasterio.open(find_file(f"VLUCD_L2_250m_{y}.tif", IN)) as src:
        stack[i] = LUT[src.read(1)]
valid = inside & (stack > 0).all(axis=0)
stack[:, ~valid] = 0
rows_, cols_ = np.flatnonzero(valid.any(1)), np.flatnonzero(valid.any(0))
r0, c0 = rows_[0], cols_[0]
stack, valid = stack[:, r0:rows_[-1] + 1, c0:cols_[-1] + 1], valid[r0:rows_[-1] + 1, c0:cols_[-1] + 1]
H, W = valid.shape
Hp, Wp = -(-H // TILE) * TILE, -(-W // TILE) * TILE
stack = np.pad(stack, ((0, 0), (0, Hp - H), (0, Wp - W)))
valid = np.pad(valid, ((0, Hp - H), (0, Wp - W)))

net = TemporalViT()
net.load_state_dict(torch.load(PRED / "vit_seed0.pt", map_location="cpu"))
net.eval()
spatial = np.full((Hp // P, Wp // P), np.nan, dtype=np.float32)
focus = np.full_like(spatial, np.nan)
year_share = []
tiles = [(r, c) for r in range(0, Hp, TILE) for c in range(0, Wp, TILE) if valid[r:r + TILE, c:c + TILE].any()]
with torch.no_grad():
    for b0 in range(0, len(tiles), 64):
        chunk = tiles[b0:b0 + 64]
        x = torch.stack([F.one_hot(torch.from_numpy(stack[:, r:r + TILE, c:c + TILE].astype(np.int64)), K + 1)
                         [..., 1:].permute(0, 3, 1, 2).float() for r, c in chunk])
        net(x, keep=True)
        rec = net.received().numpy()
        for (r, c), a in zip(chunk, rec, strict=True):
            tok_valid = valid[r:r + TILE, c:c + TILE].reshape(G, P, G, P).mean((1, 3)) > 0.5
            s = a.sum(0).reshape(G, G) * G * G
            f = (np.array(YEARS)[:, None] * a).sum(0).reshape(G, G) / a.sum(0).reshape(G, G)
            spatial[r // P:r // P + G, c // P:c // P + G] = np.where(tok_valid, s, np.nan)
            focus[r // P:r // P + G, c // P:c // P + G] = np.where(tok_valid, f, np.nan)
            if tok_valid.any():
                year_share.append(a.sum(-1))
ys = np.mean(year_share, axis=0)
ys = ys / ys.sum()

chg = ((stack != stack[0:1]).any(0) & valid).reshape(Hp // P, P, Wp // P, P).mean((1, 3))
ok = np.isfinite(spatial)
changed = ok & (chg > 0.1)
print(f"tokens: {ok.sum()}, relative attention: changed tokens {np.nanmean(spatial[changed]):.3f}, "
      f"stable tokens {np.nanmean(spatial[ok & ~changed]):.3f}; year shares {np.round(ys, 4)}")
pd.DataFrame({"input_year": YEARS, "attention_share": ys}).to_csv(OUT / "fig12_attention_data.csv", index=False,
                                                                 float_format="%.4f")

W0, E0, S0, N0 = 102.1, 109.6, 8.4, 23.45
LAT0 = 0.5 * (S0 + N0)
ext = (tr.c + c0 * tr.a, tr.c + (c0 + Wp) * tr.a, tr.f + (r0 + Hp) * tr.e, tr.f + r0 * tr.e)
pw, gx, lm, bm = 1.9, 1.05, 0.42, 0.3
ph = pw * (N0 - S0) / ((E0 - W0) * np.cos(np.radians(LAT0)))
fw, fh = lm + 2 * pw + gx + 0.55 + 2.0, bm + ph + 0.25
fig = plt.figure(figsize=(fw, fh))


def mapax(col: int, a: np.ndarray, cmap: str, norm: Normalize, title: str, label: str, fmt: str) -> Axes:
    x0 = lm + col * (pw + gx)
    ax = fig.add_axes((x0 / fw, bm / fh, pw / fw, ph / fh))
    ax.set_facecolor("#f4f4f4")
    im = ax.imshow(a, extent=ext, cmap=cmap, norm=norm, interpolation="nearest", zorder=2, rasterized=True)
    adm0.boundary.plot(ax=ax, color="#222222", lw=0.25, zorder=4)
    ax.set_xlim(W0, E0)
    ax.set_ylim(S0, N0)
    ax.set_aspect(1 / np.cos(np.radians(LAT0)))
    for xg in (104, 106, 108):
        ax.axvline(xg, color="white", lw=0.5, zorder=1.5)
    for yg in range(10, 24, 2):
        ax.axhline(yg, color="white", lw=0.5, zorder=1.5)
    ax.xaxis.set_major_locator(FixedLocator([104, 106, 108]))
    ax.yaxis.set_major_locator(FixedLocator(list(range(10, 24, 4))))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=6.5)
    ax.tick_params(which="major", length=3)
    ax.tick_params(which="minor", length=1.5)
    ax.set_title(title, fontsize=7.5, loc="left", pad=3)
    cax = fig.add_axes(((x0 + pw + 0.06) / fw, (bm + 0.15 * ph) / fh, 0.09 / fw, 0.55 * ph / fh))
    cb = fig.colorbar(im, cax=cax, format=fmt)
    if isinstance(norm, LogNorm):
        cb.set_ticks([0.5, 1, 2, 3], labels=["0.5", "1", "2", "3"])
        cb.ax.minorticks_off()
    cb.set_label(label, fontsize=6.8)
    cb.ax.tick_params(labelsize=6.3, length=2.5, direction="in")
    return ax


mapax(0, spatial, "turbo", LogNorm(0.35, 3.0), "(a) Spatial attention (2 km tokens)", "Relative attention", "%.1f")
mapax(1, focus, "turbo", Normalize(2007.3, 2008.8), "(b) Attention-weighted input year", "Mean year", "%.1f")
x0 = lm + 2 * (pw + gx) + 0.1
axb = fig.add_axes((x0 / fw, (bm + 0.45 * ph) / fh, 1.75 / fw, 0.45 * ph / fh))
bars = axb.bar([str(y) for y in YEARS], 100 * ys, color=plt.get_cmap("turbo")(np.linspace(0.15, 0.85, 5)),
               edgecolor="white", lw=0.3)
for b_, v in zip(bars, ys, strict=True):
    axb.text(b_.get_x() + b_.get_width() / 2, 100 * v + 0.3, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=6.3)
axb.set_ylim(0, 100 * ys.max() * 1.2)
axb.yaxis.set_minor_locator(MultipleLocator(1))
axb.tick_params(which="both", direction="in", right=True, labelsize=6.5)
axb.tick_params(axis="x", length=0)
axb.grid(axis="y", lw=0.4, color="0.88")
axb.set_axisbelow(True)
axb.set_ylabel("Share of attention (%)", fontsize=7)
axb.set_xlabel("Input year", fontsize=7)
axb.set_title("(c) Attention per input year", fontsize=7.5, loc="left", pad=3)
fig.canvas.draw()
rnd = fig.canvas.get_renderer()
boxes = [a.get_tightbbox(rnd) for a in fig.axes]
ov = [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if boxes[i].overlaps(boxes[j])]
print("overlapping axes (incl. labels):", ov)
fig.savefig(OUT / "fig12_attention.pdf", dpi=600)
fig.savefig(OUT / "fig12_attention.png", dpi=600)
print("range spatial:", np.nanpercentile(spatial, [1, 50, 99]).round(2), "focus:", np.nanpercentile(focus, [1, 50, 99]).round(2))
