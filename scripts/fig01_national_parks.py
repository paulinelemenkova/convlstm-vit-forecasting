from __future__ import annotations

import os
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from shapely.geometry import box

DATA = Path(os.environ.get("DATA_DIR", "./data_parks"))
OUT = Path(os.environ.get("FIG_OUT", "./figs"))
MONTH = os.environ.get("WDPA_MONTH", "Sep2026")
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "wdpa": f"https://d1gam3xoknrgr2.cloudfront.net/current/WDPA_WDOECM_{MONTH}_Public_VNM_shp.zip",
    "hdx": "https://data.humdata.org/dataset/3cb544a9-9d04-4f54-94e2-93230efd8ceb/resource/"
    "cb274bdf-f332-48bd-bfcc-d6fe3a5e4334/download/vnm_admin_boundaries.shp.zip",
    "ne": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_10m_admin_0_countries.geojson",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans", "Helvetica"])).lower()


def fetch(key: str, name: str) -> Path:
    p = DATA / name
    if not p.exists():
        urllib.request.urlretrieve(URLS[key], p)
    return p


def unzip_all(z: Path, dest: Path) -> None:
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    for inner in dest.glob("*.zip"):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(dest / inner.stem)


def ascii_name(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


wdpa_dir = DATA / "wdpa"
if not wdpa_dir.exists():
    unzip_all(fetch("wdpa", "wdpa_vnm.zip"), wdpa_dir)
hdx_dir = DATA / "hdx"
if not hdx_dir.exists():
    unzip_all(fetch("hdx", "hdx_vnm.zip"), hdx_dir)
ne = gpd.read_file(fetch("ne", "ne_10m_admin_0_countries.geojson"))

polys = pd.concat([gpd.read_file(p) for p in wdpa_dir.rglob("*polygons.shp")], ignore_index=True)
pts = pd.concat([gpd.read_file(p) for p in wdpa_dir.rglob("*points.shp")], ignore_index=True)
is_np = polys["DESIG_ENG"].str.contains("national park", case=False, na=False)
parks = gpd.GeoDataFrame(
    polys[is_np].drop_duplicates("SITE_ID" if "SITE_ID" in polys else "WDPAID"), crs=polys.crs
)
n_pts = int(pts["DESIG_ENG"].str.contains("national park", case=False, na=False).sum())
parks = parks.to_crs(4326)
parks["label_pt"] = parks.geometry.representative_point()
parks["lat"] = parks.label_pt.y
parks = parks.sort_values("lat", ascending=False).reset_index(drop=True)
parks["no"] = np.arange(1, len(parks) + 1)
parks["short"] = (
    parks["NAME"]
    .str.replace("Vườn quốc gia ", "", regex=False)
    .str.replace(" - ", "–")
    .str.replace("-", "–")
    .str.replace("– ", "–")
    .map(ascii_name)
)
print(f"national parks: {len(parks)} polygons, {n_pts} point records")

adm0 = gpd.read_file(hdx_dir.rglob("vnm_admin0.shp").__next__()).to_crs(4326)
adm1 = gpd.read_file(hdx_dir.rglob("vnm_admin1.shp").__next__()).to_crs(4326)

W, E, S, N = 102.0, 109.8, 8.3, 23.5
LAT0 = 0.5 * (S + N)
map_w = 3.25
map_h = map_w * (N - S) / ((E - W) * np.cos(np.radians(LAT0)))
key_w = 2.05
fig = plt.figure(figsize=(map_w + key_w + 0.35, map_h + 0.3))
fw, fh = fig.get_size_inches()
ax = fig.add_axes((0.35 / fw, 0.25 / fh, map_w / fw, map_h / fh))
axk = fig.add_axes(((0.35 + map_w + 0.08) / fw, 0.25 / fh, (key_w - 0.08) / fw, map_h / fh))
axk.axis("off")
ax.set_xlim(W, E)
ax.set_ylim(S, N)
ax.set_facecolor("#dfe9f1")

frame = box(W - 1, S - 1, E + 1, N + 1)
ne[ne.intersects(frame) & (ne["ADM0_A3"] != "VNM")].plot(
    ax=ax, color="#ebe9e4", edgecolor="#9a9a9a", lw=0.3, zorder=1
)
adm0.plot(ax=ax, color="#f7f4ea", edgecolor="none", zorder=2)
adm1.boundary.plot(ax=ax, color="#9b9b9b", lw=0.25, zorder=3)
adm0.boundary.plot(ax=ax, color="#222222", lw=0.6, zorder=4)
parks.plot(ax=ax, color="#2e8b3d", edgecolor="#135a22", lw=0.4, alpha=0.9, zorder=5)

for x in np.arange(102, 110, 2):
    ax.axvline(x, color="white", lw=0.5, zorder=4.5)
for y in np.arange(10, 24, 2):
    ax.axhline(y, color="white", lw=0.5, zorder=4.5)
ax.xaxis.set_major_locator(FixedLocator(np.arange(102, 110, 2)))
ax.yaxis.set_major_locator(FixedLocator(np.arange(10, 24, 2)))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°E"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=7.5)
ax.tick_params(which="major", length=4)
ax.tick_params(which="minor", length=2)

halo = [withStroke(linewidth=1.8, foreground="white")]
fig.canvas.draw()
rnd = fig.canvas.get_renderer()
placed: list = []
labels = []
cands = [(0.0, 0.0)] + [
    (r * np.cos(a), r * np.sin(a))
    for r in (0.28, 0.42, 0.58, 0.75)
    for a in np.radians([0, 180, 45, 135, 315, 225, 90, 270])
]
for _, p in parks.iterrows():
    px, py = p.label_pt.x, p.label_pt.y
    for dx, dy in cands:
        t = ax.text(
            px + dx,
            py + dy,
            str(p.no),
            ha="center",
            va="center",
            fontsize=6.5,
            weight="bold",
            color="#0d3d17",
            path_effects=halo,
            zorder=8,
        )
        bb = t.get_window_extent(rnd).expanded(1.08, 1.12)
        inside = ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
        if inside and not any(bb.overlaps(q) for q in placed):
            placed.append(bb)
            labels.append(t)
            if (dx, dy) != (0.0, 0.0):
                ax.plot([px, px + dx * 0.75], [py, py + dy * 0.75], color="#0d3d17", lw=0.4, zorder=7)
            break
        t.remove()
    else:
        raise RuntimeError(f"no free label position for park {p.no}")

km_deg = 111.32 * np.cos(np.radians(9.0))
x0, y0 = 107.3, 8.75
L = 200 / km_deg
ax.plot([x0, x0 + L / 2], [y0, y0], color="k", lw=2.4, solid_capstyle="butt", zorder=9)
ax.plot([x0 + L / 2, x0 + L], [y0, y0], color="k", lw=2.4, solid_capstyle="butt", zorder=9)
ax.plot([x0 + L / 2, x0 + L], [y0, y0], color="white", lw=1.3, solid_capstyle="butt", zorder=9.5)
for xv, s in ((x0, "0"), (x0 + L, "200 km")):
    ax.text(xv, y0 + 0.2, s, ha="center", va="bottom", fontsize=7, path_effects=halo, zorder=9)
ax.annotate(
    "N",
    xy=(109.35, 23.15),
    xytext=(109.35, 22.35),
    ha="center",
    va="center",
    fontsize=8,
    weight="bold",
    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0),
    zorder=9,
)

axk.set_xlim(0, 1)
axk.set_ylim(0, 1)
axk.text(0.0, 1.0, "National parks (WDPA)", fontsize=7.5, weight="bold", va="top")
step = 0.94 / max(len(parks), 1)
for i, p in parks.iterrows():
    y = 0.955 - i * step
    axk.text(0.0, y, f"{p.no}", fontsize=6.5, weight="bold", ha="left", va="top", color="#0d3d17")
    axk.text(0.09, y, f"{p.short} ({p.REP_AREA:.0f} km²)", fontsize=6.5, ha="left", va="top")

fig.canvas.draw()
bbs = [t.get_window_extent(rnd) for t in labels]
ov = sum(1 for i in range(len(bbs)) for j in range(i + 1, len(bbs)) if bbs[i].overlaps(bbs[j]))
key_bottom = axk.transAxes.inverted().transform(axk.texts[-1].get_window_extent(rnd).p0)[1]
print(f"labels: {len(labels)}, overlaps: {ov}, key bottom (axes fraction): {key_bottom:.3f}")
print(parks[["no", "short", "REP_AREA", "STATUS_YR"]].to_string(index=False))
fig.savefig(OUT / "fig01_national_parks.pdf")
fig.savefig(OUT / "fig01_national_parks.png", dpi=600)
print("saved")
