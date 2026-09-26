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
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from rasterio.features import rasterize

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

GROUPS = {1: ("Natural forest", [10, 11, 12, 20], "#1b6e2d"), 2: ("Plantation", [14], "#8fce5a"),
          3: ("Mangrove", [15], "#b24fd1"), 4: ("Woody crops", [4], "#d9b25f"),
          5: ("Other agriculture", [3, 5, 6], "#3a86a7"), 6: ("Built-up", [1, 2], "#c0392b"),
          7: ("Other land", [7, 8, 9, 16, 18, 19], "#a3a3a3")}
AGREE = [("Both: no change from 2020", "#e8e4da"), ("Both: same new class", "#2c7bb6"),
         ("Models disagree", "#e66101")]
LUT = np.zeros(256, dtype=np.uint8)
for gid, (_n, codes, _c) in GROUPS.items():
    LUT[codes] = gid
STEP = 2

with rasterio.open(find_file("VLUCD_L2_250m_2020.tif", IN)) as src:
    o20 = LUT[src.read(1)]
    tr, crs, shape = src.transform, src.crs, src.shape
adm0 = gpd.read_file(ADM0).to_crs(crs)
valid = rasterize(((g, 1) for g in adm0.geometry), out_shape=shape, transform=tr, dtype="uint8") == 1
with rasterio.open(PRED / "convlstm_2030.tif") as src:
    cl = src.read(1)
with rasterio.open(PRED / "vit_2030.tif") as src:
    vt = src.read(1)
valid &= (o20 > 0) & (cl > 0) & (vt > 0)
o20, cl, vt = (np.where(valid, a, 0) for a in (o20, cl, vt))
agree = np.zeros(shape, dtype=np.uint8)
agree[valid & (cl == vt) & (cl == o20)] = 1
agree[valid & (cl == vt) & (cl != o20)] = 2
agree[valid & (cl != vt)] = 3

r = 6371.0072
lat = tr.f + np.arange(shape[0]) * tr.e
row_area = r**2 * np.radians(abs(tr.a)) * np.abs(np.sin(np.radians(lat)) - np.sin(np.radians(lat + tr.e)))
w = np.broadcast_to(row_area[:, None], shape)
share = {n: 100 * w[agree == k + 1].sum() / w[valid].sum() for k, (n, _c) in enumerate(AGREE)}
pd.Series(share, name="share_pct").to_csv(OUT / "fig11_agreement_2030.csv", float_format="%.2f")

W, E, S, N = 102.1, 109.6, 8.4, 23.45
LAT0 = 0.5 * (S + N)
pw, gx, lm, bm, tm = 1.55, 0.12, 0.42, 1.0, 0.22
ph = pw * (N - S) / ((E - W) * np.cos(np.radians(LAT0)))
fw, fh = lm + 4 * pw + 3 * gx + 0.05, bm + ph + tm
fig = plt.figure(figsize=(fw, fh))
ext = (tr.c, tr.c + tr.a * shape[1], tr.f + tr.e * shape[0], tr.f)
halo = [withStroke(linewidth=1.8, foreground="white")]


def panel(col: int, a: np.ndarray, colors: list[str], title: str) -> Axes:
    ax = fig.add_axes(((lm + col * (pw + gx)) / fw, bm / fh, pw / fw, ph / fh))
    n = len(colors) + 1
    ax.set_facecolor("#f4f4f4")
    ax.imshow(np.ma.masked_equal(a[::STEP, ::STEP], 0), extent=ext, cmap=ListedColormap(["#ffffff", *colors]),
              norm=BoundaryNorm(np.arange(-0.5, n + 0.5), n), interpolation="nearest", zorder=2, rasterized=True)
    adm0.boundary.plot(ax=ax, color="#333333", lw=0.2, zorder=4)
    ax.set_xlim(W, E)
    ax.set_ylim(S, N)
    ax.set_aspect(1 / np.cos(np.radians(LAT0)))
    for x in (104, 106, 108):
        ax.axvline(x, color="white", lw=0.5, zorder=1.5)
    for y in range(10, 24, 2):
        ax.axhline(y, color="white", lw=0.5, zorder=1.5)
    ax.xaxis.set_major_locator(FixedLocator([104, 106, 108]))
    ax.yaxis.set_major_locator(FixedLocator(list(range(10, 24, 4))))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=6.5, labelleft=col == 0)
    ax.tick_params(which="major", length=3)
    ax.tick_params(which="minor", length=1.5)
    ax.set_title(title, fontsize=7.5, loc="left", pad=3)
    return ax


lc = [v[2] for v in GROUPS.values()]
ax0 = panel(0, o20, lc, "(a) Observed 2020")
panel(1, cl, lc, "(b) ConvLSTM 2030")
panel(2, vt, lc, "(c) Temporal ViT 2030")
ax3 = panel(3, agree, [c for _n, c in AGREE], "(d) Agreement 2030")
ax3.text(0.04, 0.03, "\n".join(f"{v:.1f}%" for v in share.values()), transform=ax3.transAxes, fontsize=6.3,
         va="bottom", ha="left", path_effects=halo, zorder=9, linespacing=1.25)
km_deg = 111.32 * np.cos(np.radians(9.0))
x0, y0, L = 106.1, 8.85, 200 / km_deg
ax0.plot([x0, x0 + L / 2], [y0, y0], color="k", lw=2.0, solid_capstyle="butt", zorder=9)
ax0.plot([x0 + L / 2, x0 + L], [y0, y0], color="k", lw=2.0, solid_capstyle="butt", zorder=9)
ax0.plot([x0 + L / 2, x0 + L], [y0, y0], color="white", lw=1.0, solid_capstyle="butt", zorder=9.5)
for xv, s in ((x0, "0"), (x0 + L, "200 km")):
    ax0.text(xv, y0 + 0.3, s, ha="center", va="bottom", fontsize=6.5, path_effects=halo, zorder=9)
ax0.annotate("N", xy=(103.0, 16.4), xytext=(103.0, 15.4), ha="center", va="center", fontsize=7.5, weight="bold",
             arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9), zorder=9)

lg1 = fig.legend([Patch(fc=v[2], ec="0.3", lw=0.3) for v in GROUPS.values()], [v[0] for v in GROUPS.values()],
                 loc="upper left", bbox_to_anchor=(lm / fw, (bm - 0.3) / fh), ncol=4, fontsize=6.5, frameon=False,
                 handlelength=1.2, handleheight=0.9, columnspacing=1.0, title="Class group (a–c)", title_fontsize=7,
                 alignment="left")
lg2 = fig.legend([Patch(fc=c, ec="0.3", lw=0.3) for _n, c in AGREE],
                 [f"{n} ({v:.1f}%)" for (n, _c), v in zip(AGREE, share.values(), strict=True)],
                 loc="upper right", bbox_to_anchor=(1.0, (bm - 0.3) / fh), ncol=1, fontsize=6.5, frameon=False,
                 handlelength=1.2, handleheight=0.9, title="Agreement (d)", title_fontsize=7, alignment="left")
for lg in (lg1, lg2):
    lg.get_title().set_weight("bold")
ax3.texts[-1].remove()

fig.canvas.draw()
rnd = fig.canvas.get_renderer()
b1, b2 = lg1.get_window_extent(rnd), lg2.get_window_extent(rnd)
assert not b1.overlaps(b2), "legends overlap"
lowest_tick = min(a.get_tightbbox(rnd).y0 for a in fig.axes)
assert max(b1.y1, b2.y1) < lowest_tick, "legend overlaps tick labels"
print({k: round(v, 2) for k, v in share.items()})
fig.savefig(OUT / "fig11_forecast_2030.pdf", dpi=600)
fig.savefig(OUT / "fig11_forecast_2030.png", dpi=600)
