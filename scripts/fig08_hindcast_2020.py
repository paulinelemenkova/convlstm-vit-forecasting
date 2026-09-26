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
PRED = Path(os.environ.get("PRED_DIR", "./hindcast"))
OUT = Path(os.environ.get("FIG_OUT", "./figs"))
OUT.mkdir(parents=True, exist_ok=True)


def find_file(name: str, root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"folder does not exist: {root} (check the spelling; Finder may hide an extension such as .shp)")
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(
            f"{name} not found below {root.resolve()}. Set JAXA_DIR to the folder with the unzipped JAXA maps and "
            f"HDX_DIR to the folder with vnm_admin0.shp and vnm_admin1.shp (both are searched recursively), e.g.\n"
            f"  JAXA_DIR=~/data/JAXA HDX_DIR=~/data/HDX python3 {Path(__file__).name}")
    return hits[0]


ADM0 = ADM0 if ADM0.is_file() else find_file("vnm_admin0.shp", Path(os.environ.get("HDX_DIR", ".")).expanduser())


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans", "Helvetica"])).lower()

GROUPS = {
    1: ("Natural forest", [10, 11, 12, 20], "#1b6e2d"),
    2: ("Plantation", [14], "#8fce5a"),
    3: ("Mangrove", [15], "#b24fd1"),
    4: ("Woody crops", [4], "#d9b25f"),
    5: ("Other agriculture", [3, 5, 6], "#3a86a7"),
    6: ("Built-up", [1, 2], "#c0392b"),
    7: ("Other land", [7, 8, 9, 16, 18, 19], "#a3a3a3"),
}
ERRORS = [
    ("Correct persistence", "#e8e4da"),
    ("Hit (correct change)", "#1a9641"),
    ("Wrong class of change", "#fdae61"),
    ("Miss (change not predicted)", "#d7191c"),
    ("False alarm", "#2c7bb6"),
]
MODELS = [
    ("Persistence", "persistence"),
    ("CA–Markov", "camarkov"),
    ("Random Forest", "rf"),
    ("ConvLSTM", "convlstm"),
    ("Temporal ViT", "vit"),
]
STEP = 2

LUT = np.zeros(256, dtype=np.uint8)
for gid, (_gname, codes, _col) in GROUPS.items():
    LUT[codes] = gid


def read(p: Path, lut: bool) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(p) as src:
        a = src.read(1)
        return (LUT[a] if lut else a), src.transform, src.crs


o10, transform, crs = read(find_file("VLUCD_L2_250m_2010.tif", IN), True)
o20, _t, _c = read(find_file("VLUCD_L2_250m_2020.tif", IN), True)
adm0 = gpd.read_file(ADM0).to_crs(crs)
inside = rasterize(((g, 1) for g in adm0.geometry), out_shape=o10.shape, transform=transform, dtype="uint8") == 1

preds: dict[str, np.ndarray] = {}
for label, stem in MODELS:
    f = PRED / f"{stem}_2020.tif"
    if f.exists():
        preds[label] = read(f, False)[0]
assert "Persistence" in preds, "run 03_hindcast_baselines.py first"
valid = inside & (o10 > 0) & (o20 > 0) & (preds["Persistence"] > 0)
for p in preds.values():
    valid &= p > 0
o10 = np.where(valid, o10, 0)
o20 = np.where(valid, o20, 0)
met = pd.read_csv(PRED / "hindcast_metrics.csv").set_index("Model")


def errors(p: np.ndarray) -> np.ndarray:
    e = np.zeros(p.shape, dtype=np.uint8)
    ch_o, ch_p = o20 != o10, p != o10
    e[valid & ~ch_o & ~ch_p] = 1
    e[valid & ch_o & ch_p & (p == o20)] = 2
    e[valid & ch_o & ch_p & (p != o20)] = 3
    e[valid & ch_o & ~ch_p] = 4
    e[valid & ~ch_o & ch_p] = 5
    return e


W, E, S, N = 102.1, 109.6, 8.4, 23.45
LAT0 = 0.5 * (S + N)
others = [m for m in preds if m != "Persistence"]
ncol = 2 + len(others)
pw = 1.5 if ncol <= 4 else 1.12
ph = pw * (N - S) / ((E - W) * np.cos(np.radians(LAT0)))
gx, gy, lm, bm, tm = 0.1, 0.34, 0.42, 0.3, 0.22
fw = lm + ncol * pw + (ncol - 1) * gx + 0.05
fh = bm + 2 * ph + gy + tm
fig = plt.figure(figsize=(fw, fh))


def panel(row: int, col: int) -> Axes:
    x = lm + col * (pw + gx)
    y = bm + (1 - row) * (ph + gy)
    return fig.add_axes((x / fw, y / fh, pw / fw, ph / fh))


ext = (transform.c, transform.c + transform.a * o10.shape[1], transform.f + transform.e * o10.shape[0], transform.f)
cm_lc = ListedColormap(["#ffffff"] + [v[2] for v in GROUPS.values()])
cm_er = ListedColormap(["#ffffff"] + [c for _, c in ERRORS])
halo = [withStroke(linewidth=1.8, foreground="white")]


def show(ax: Axes, a: np.ndarray, cmap: ListedColormap, title: str, row: int, col: int) -> None:
    n = cmap.N
    ax.set_facecolor("#f4f4f4")
    ax.imshow(np.ma.masked_equal(a[::STEP, ::STEP], 0), extent=ext, cmap=cmap,
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
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=6.5,
                   labelbottom=row == 1, labelleft=col == 0)
    ax.tick_params(which="major", length=3)
    ax.tick_params(which="minor", length=1.5)
    ax.set_title(title, fontsize=fs_title, loc="left", pad=3)


letters = iter("abcdefghijklmn")
SHORT = {"Random Forest": "RF", "Temporal ViT": "ViT"} if ncol > 4 else {}
fs_title = 7.5 if ncol <= 4 else 6.8


def mtxt(m: str) -> str:
    return f"OA {met.loc[m, 'OA']:.3f}\nFoM {met.loc[m, 'FoM']:.3f}" if m in met.index else ""


top = [("Observed 2010", o10), ("Observed 2020", o20)] + [(f"{SHORT.get(m, m)} 2020", preds[m]) for m in others]
axes_top = []
for c, (t, a) in enumerate(top):
    ax = panel(0, c)
    show(ax, a, cm_lc, f"({next(letters)}) {t}", 0, c)
    axes_top.append(ax)

bottom = [("Persistence", 0)] + [(m, 2 + i) for i, m in enumerate(others)]
for m, c in bottom:
    ax = panel(1, c)
    show(ax, errors(preds[m]), cm_er, f"({next(letters)}) {SHORT.get(m, m)} errors", 1, c)
    ax.text(0.04, 0.03, mtxt(m), transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, zorder=9,
            path_effects=halo, linespacing=1.2)

ax = axes_top[0]
km_deg = 111.32 * np.cos(np.radians(9.0))
x0, y0, L = 106.1, 8.85, 200 / km_deg
ax.plot([x0, x0 + L / 2], [y0, y0], color="k", lw=2.0, solid_capstyle="butt", zorder=9)
ax.plot([x0 + L / 2, x0 + L], [y0, y0], color="k", lw=2.0, solid_capstyle="butt", zorder=9)
ax.plot([x0 + L / 2, x0 + L], [y0, y0], color="white", lw=1.0, solid_capstyle="butt", zorder=9.5)
for xv, s in ((x0, "0"), (x0 + L, "200 km")):
    ax.text(xv, y0 + 0.3, s, ha="center", va="bottom", fontsize=6.5, path_effects=halo, zorder=9)
ax.annotate("N", xy=(103.0, 16.4), xytext=(103.0, 15.4), ha="center", va="center", fontsize=7.5, weight="bold",
            arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9), zorder=9)

axl = panel(1, 1)
axl.axis("off")
fs_leg = 6.5 if ncol <= 4 else 6.0
err_lab = [n for n, _ in ERRORS] if ncol <= 4 else ["Correct persistence", "Hit", "Wrong class", "Miss", "False alarm"]
lg1 = axl.legend([Patch(fc=v[2], ec="0.3", lw=0.3) for v in GROUPS.values()], [v[0] for v in GROUPS.values()],
                 loc="upper left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=fs_leg, title="Class group",
                 title_fontsize=7, alignment="left", handlelength=1.2, handleheight=0.9, borderaxespad=0.0)
lg1.get_title().set_weight("bold")
axl.add_artist(lg1)
fig.canvas.draw()
y_next = axl.transAxes.inverted().transform((0, lg1.get_window_extent(fig.canvas.get_renderer()).y0))[1] - 0.06
lg2 = axl.legend([Patch(fc=c, ec="0.3", lw=0.3) for _, c in ERRORS], err_lab, loc="upper left",
                 bbox_to_anchor=(-0.02, y_next), frameon=False, fontsize=fs_leg, title="Change 2010–2020",
                 title_fontsize=7, alignment="left", handlelength=1.2, handleheight=0.9, borderaxespad=0.0)
lg2.get_title().set_weight("bold")

fig.canvas.draw()
rnd = fig.canvas.get_renderer()
b1, b2 = lg1.get_window_extent(rnd), lg2.get_window_extent(rnd)
fr = fig.bbox
assert not b1.overlaps(b2), "legends overlap"
for a_ in fig.axes:
    if a_.get_title(loc="left"):
        tb = a_.title.get_window_extent(rnd) if a_.title.get_text() else a_._left_title.get_window_extent(rnd)
        assert tb.x1 <= a_.bbox.x1 + 1, f"title wider than panel: {a_.get_title(loc='left')}"
assert b2.y0 >= axl.bbox.y0 - 2 and max(b1.x1, b2.x1) <= axl.bbox.x1 + gx * fig.dpi, "legend outside its cell"
assert all(fr.contains(b.x1, b.y1) for b in (b1, b2))
print("models:", list(preds), "| figure size (in):", round(fw, 2), round(fh, 2))
fig.savefig(OUT / "fig08_hindcast_2020.pdf", dpi=600)
fig.savefig(OUT / "fig08_hindcast_2020.png", dpi=600)
