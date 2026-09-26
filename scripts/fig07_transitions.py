from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from rasterio.features import rasterize

IN = Path(os.environ.get("JAXA_DIR", ".")).expanduser()
ADM0 = Path(os.environ.get("VN_ADMIN0", "")).expanduser()
ADM1 = Path(os.environ.get("VN_ADMIN1", "")).expanduser()
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
ADM1 = ADM1 if ADM1.is_file() else find_file("vnm_admin1.shp", Path(os.environ.get("HDX_DIR", ".")).expanduser())


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans", "Helvetica"])).lower()

GROUPS: dict[int, tuple[str, list[int]]] = {
    1: ("Natural forest", [10, 11, 12, 20]),
    2: ("Plantation", [14]),
    3: ("Mangrove", [15]),
    4: ("Woody crops", [4]),
    5: ("Other agriculture", [3, 5, 6]),
    6: ("Built-up", [1, 2]),
    7: ("Other land", [7, 8, 9, 16, 18, 19]),
}
LUT = np.zeros(256, dtype=np.uint8)
for gid, (_gname, codes) in GROUPS.items():
    LUT[codes] = gid
FOREST = (1, 2, 3)

CHANGE = [
    ("Stable forest", "#1b6e2d"),
    ("Change of forest type", "#a6d96a"),
    ("Forest loss", "#d7301f"),
    ("Forest gain", "#2c7bb6"),
    ("Stable non-forest", "#e6e1d3"),
]


def read_groups(year: int) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    f = find_file(f"VLUCD_L2_250m_{year}.tif", IN)
    with rasterio.open(f) as src:
        return LUT[src.read(1)], src.transform, src.crs


def pixel_area_km2(transform: rasterio.Affine, nrows: int) -> np.ndarray:
    r = 6371.0072
    lat_top = transform.f + np.arange(nrows) * transform.e
    lat_bot = lat_top + transform.e
    return r**2 * np.radians(abs(transform.a)) * np.abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot)))


g0, transform, crs = read_groups(1990)
g1, _t2020, _c2020 = read_groups(2020)
adm0 = gpd.read_file(ADM0).to_crs(crs)
adm1 = gpd.read_file(ADM1).to_crs(crs)
inside = rasterize(((g, 1) for g in adm0.geometry), out_shape=g0.shape, transform=transform, dtype="uint8") == 1
valid = inside & (g0 > 0) & (g1 > 0)
f0, f1 = np.isin(g0, FOREST), np.isin(g1, FOREST)

chg = np.zeros(g0.shape, dtype=np.uint8)
chg[valid & f0 & f1 & (g0 == g1)] = 1
chg[valid & f0 & f1 & (g0 != g1)] = 2
chg[valid & f0 & ~f1] = 3
chg[valid & ~f0 & f1] = 4
chg[valid & ~f0 & ~f1] = 5

area = np.broadcast_to(pixel_area_km2(transform, g0.shape[0])[:, None], g0.shape)
chg_area = np.bincount(chg.ravel(), weights=area.ravel(), minlength=6)[1:]
mat = np.bincount((g0[valid].astype(int) - 1) * 7 + (g1[valid] - 1), weights=area[valid], minlength=49)
mat = mat.reshape(7, 7) / 1000.0

W, E, S, N = 102.1, 110.3, 8.4, 23.45
LAT0 = 0.5 * (S + N)
map_w = 3.3
map_h = map_w * (N - S) / ((E - W) * np.cos(np.radians(LAT0)))
right_w = 2.55
gap = 1.05
fig = plt.figure(figsize=(0.4 + map_w + gap + right_w + 0.3, map_h + 0.35))
fw, fh = fig.get_size_inches()
ax = fig.add_axes((0.4 / fw, 0.3 / fh, map_w / fw, map_h / fh))
mx0 = 0.4 + map_w + gap
mat_side = right_w
mat_top = 0.3 + map_h - 1.55
axm = fig.add_axes((mx0 / fw, (mat_top - mat_side) / fh, mat_side / fw, mat_side / fh))

ax.set_facecolor("#f4f4f4")
ext = (transform.c, transform.c + transform.a * g0.shape[1], transform.f + transform.e * g0.shape[0], transform.f)
rgba = ListedColormap(["#ffffff"] + [c for _, c in CHANGE])
img = np.ma.masked_equal(chg, 0)
ax.imshow(img, extent=ext, cmap=rgba, norm=BoundaryNorm(np.arange(-0.5, 6.5), 6), interpolation="nearest",
          zorder=2, rasterized=True)
adm1.boundary.plot(ax=ax, color="#555555", lw=0.25, zorder=3)
adm0.boundary.plot(ax=ax, color="#111111", lw=0.4, zorder=4)
ax.set_xlim(W, E)
ax.set_ylim(S, N)
ax.set_aspect(1 / np.cos(np.radians(LAT0)))
for x in np.arange(104, 112, 2):
    ax.axvline(x, color="white", lw=0.6, zorder=1.5)
for y in np.arange(10, 24, 2):
    ax.axhline(y, color="white", lw=0.5, zorder=1.5)
ax.xaxis.set_major_locator(FixedLocator(np.arange(104, 112, 2)))
ax.yaxis.set_major_locator(FixedLocator(np.arange(10, 24, 2)))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°E"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=7.5)
ax.tick_params(which="major", length=4)
ax.tick_params(which="minor", length=2)
ax.text(0.03, 0.985, "(a)", transform=ax.transAxes, ha="left", va="top", weight="bold", fontsize=10, zorder=9)

halo = [withStroke(linewidth=2.0, foreground="white")]
LABELS = {
    "Son La": (102.75, 20.2),
    "Lang Son": (108.2, 22.6),
    "Gia Lai": (109.75, 14.6),
    "Dak Lak": (109.65, 13.3),
    "Lam Dong": (109.5, 10.9),
}
adm1_ll = adm1.to_crs(4326)
lab_boxes = []
fig.canvas.draw()
rnd = fig.canvas.get_renderer()
for name, (lx, ly) in LABELS.items():
    row = adm1_ll[adm1_ll["adm1_name"] == name]
    assert len(row) == 1, name
    p = row.geometry.iloc[0].representative_point()
    ax.plot(p.x, p.y, "o", ms=2.4, mfc="k", mec="white", mew=0.5, zorder=8)
    t = ax.annotate(name, xy=(p.x, p.y), xytext=(lx, ly), ha="center", va="center", fontsize=7, zorder=9,
                    path_effects=halo, arrowprops=dict(arrowstyle="-", color="k", lw=0.5, shrinkA=1.5, shrinkB=1.5))
    lab_boxes.append(t.get_window_extent(rnd))

km_deg = 111.32 * np.cos(np.radians(9.0))
x0, y0 = 107.6, 8.75
L = 200 / km_deg
ax.plot([x0, x0 + L / 2], [y0, y0], color="k", lw=2.4, solid_capstyle="butt", zorder=9)
ax.plot([x0 + L / 2, x0 + L], [y0, y0], color="k", lw=2.4, solid_capstyle="butt", zorder=9)
ax.plot([x0 + L / 2, x0 + L], [y0, y0], color="white", lw=1.3, solid_capstyle="butt", zorder=9.5)
for xv, s in ((x0, "0"), (x0 + L, "200 km")):
    ax.text(xv, y0 + 0.2, s, ha="center", va="bottom", fontsize=7, path_effects=halo, zorder=9)
ax.annotate("N", xy=(109.9, 23.2), xytext=(109.9, 22.45), ha="center", va="center", fontsize=8, weight="bold",
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0), zorder=9)

names = [GROUPS[k][0] for k in range(1, 8)]
short = ["Natural forest", "Plantation", "Mangrove", "Woody crops", "Other agric.", "Built-up", "Other land"]
cmap = plt.get_cmap("turbo")
norm = LogNorm(vmin=0.1, vmax=mat.max())
axm.imshow(np.where(mat > 0.05, mat, np.nan), cmap=cmap, norm=norm, aspect="equal")
axm.set_facecolor("#f2f2f2")
for i in range(7):
    for j in range(7):
        v = mat[i, j]
        txt = f"{v:.1f}" if v >= 0.05 else "<0.1" if v > 0 else "0"
        rgb = np.array(cmap(norm(max(v, 0.1)))[:3]) if v >= 0.05 else np.array([0.95, 0.95, 0.95])
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        axm.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                 color="white" if lum < 0.45 else "black", weight="bold" if i == j else "normal")
for k in range(8):
    axm.axhline(k - 0.5, color="white", lw=1.0)
    axm.axvline(k - 0.5, color="white", lw=1.0)
axm.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="k", lw=1.2, zorder=5))
axm.set_xticks(range(7), short, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
axm.set_yticks(range(7), short, fontsize=7)
axm.tick_params(length=0)
axm.set_xlabel("Class group in 2020", fontsize=8)
axm.set_ylabel("Class group in 1990", fontsize=8)
for sp in axm.spines.values():
    sp.set_visible(False)
axm.text(-0.02, 1.02, "(b)", transform=axm.transAxes, ha="right", va="bottom", weight="bold", fontsize=10)
axm.set_title("Transitions 1990–2020 (10$^3$ km$^2$)", fontsize=8, loc="left", pad=4)

pos = axm.get_position()
cax = fig.add_axes((pos.x1 + 0.006, pos.y0, 0.012, pos.height))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
cb.ax.tick_params(labelsize=7, length=3, direction="in")
cb.outline.set_linewidth(0.5)

handles = [Patch(fc=c, ec="0.3", lw=0.4) for _, c in CHANGE]
labels = [f"{n} ({a / 1000:.1f})" for (n, _), a in zip(CHANGE, chg_area, strict=True)]
leg_y = (0.3 + map_h) / fh
leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(mx0 / fw - 0.01, leg_y), frameon=False,
                 fontsize=7.5, title="(a) Forest change 1990–2020 (10$^3$ km$^2$)", title_fontsize=8,
                 handlelength=1.4, handleheight=1.0, borderaxespad=0.0, alignment="left")
leg.get_title().set_weight("bold")

fig.canvas.draw()
ov = sum(1 for i in range(len(lab_boxes)) for j in range(i + 1, len(lab_boxes)) if lab_boxes[i].overlaps(lab_boxes[j]))
lb = leg.get_window_extent(rnd)
mt = axm.get_tightbbox(rnd)
print(f"label overlaps: {ov}; legend bottom {lb.y0:.0f} px vs matrix top {mt.y1:.0f} px (must be >)")
assert ov == 0 and lb.y0 > mt.y1
assert all(ax.bbox.contains(b.x0, b.y0) and ax.bbox.contains(b.x1, b.y1) for b in lab_boxes)
for (n, _), a in zip(CHANGE, chg_area, strict=True):
    print(f"{n:24s} {a:10.0f} km2")
print("forest 1990 / 2020 (km2):", round(area[valid & f0].sum()), round(area[valid & f1].sum()))
fig.savefig(OUT / "fig07_transitions.pdf", dpi=600)
fig.savefig(OUT / "fig07_transitions.png", dpi=600)
