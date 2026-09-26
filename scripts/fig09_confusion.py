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
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
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

GROUPS = {1: ("Natural forest", [10, 11, 12, 20]), 2: ("Plantation", [14]), 3: ("Mangrove", [15]),
          4: ("Woody crops", [4]), 5: ("Other agriculture", [3, 5, 6]), 6: ("Built-up", [1, 2]),
          7: ("Other land", [7, 8, 9, 16, 18, 19])}
SHORT = ["Nat. forest", "Plantation", "Mangrove", "Woody crops", "Other agric.", "Built-up", "Other land"]
K = 7
LUT = np.zeros(256, dtype=np.uint8)
for gid, (_n, codes) in GROUPS.items():
    LUT[codes] = gid
MODELS = [("Persistence", "persistence", "#7f7f7f", "o"), ("CA–Markov", "camarkov", "#9467bd", "s"),
          ("Random Forest", "rf", "#2ca02c", "^"), ("ConvLSTM", "convlstm", "#1f77b4", "D"),
          ("Temporal ViT", "vit", "#d62728", "v")]

with rasterio.open(find_file("VLUCD_L2_250m_2020.tif", IN)) as src:
    obs = LUT[src.read(1)]
    tr, crs, shape = src.transform, src.crs, src.shape
inside = rasterize(((g, 1) for g in gpd.read_file(ADM0).to_crs(crs).geometry), out_shape=shape, transform=tr,
                   dtype="uint8") == 1
preds = {}
for name, stem, _c, _m in MODELS:
    with rasterio.open(PRED / f"{stem}_2020.tif") as src:
        preds[name] = src.read(1)
valid = inside & (obs > 0)
for p in preds.values():
    valid &= p > 0
r = 6371.0072
lat = tr.f + np.arange(shape[0]) * tr.e
row_area = r**2 * np.radians(abs(tr.a)) * np.abs(np.sin(np.radians(lat)) - np.sin(np.radians(lat + tr.e)))
w = np.broadcast_to(row_area[:, None], shape)[valid]
o = obs[valid].astype(int) - 1

cms, f1, rows = {}, {}, []
for name, _s, _c, _m in MODELS:
    p = preds[name][valid].astype(int) - 1
    cm = np.bincount(o * K + p, weights=w, minlength=K * K).reshape(K, K)
    cms[name] = 100 * cm / cm.sum(axis=1, keepdims=True)
    f1[name] = 2 * np.diag(cm) / (cm.sum(0) + cm.sum(1))
    for i in range(K):
        for j in range(K):
            rows.append({"model": name, "observed": SHORT[i], "predicted": SHORT[j], "row_pct": cms[name][i, j],
                         "area_km2": cm[i, j]})
pd.DataFrame(rows).to_csv(OUT / "fig09_confusion_data.csv", index=False, float_format="%.3f")

fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.3), layout="constrained")
cmap, norm = plt.get_cmap("turbo"), Normalize(0, 100)
letters = "abcdef"
for k, (name, _s, _c, _m) in enumerate(MODELS):
    ax = axs.flat[k]
    m = cms[name]
    ax.imshow(m, cmap=cmap, norm=norm)
    for i in range(K):
        for j in range(K):
            rgb = cmap(norm(m[i, j]))[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=5.8,
                    color="white" if lum < 0.45 else "black", weight="bold" if i == j else "normal")
    for q in range(K + 1):
        ax.axhline(q - 0.5, color="white", lw=0.6)
        ax.axvline(q - 0.5, color="white", lw=0.6)
    ax.set_xticks(range(K), SHORT, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.3)
    ax.set_yticks(range(K), SHORT, fontsize=6.3)
    ax.tick_params(length=0, labelleft=k % 3 == 0, labelbottom=k >= 3 or k == 2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"({letters[k]}) {name}", fontsize=7.5, loc="left", pad=3)
    if k % 3 == 0:
        ax.set_ylabel("Observed 2020", fontsize=7)
    if k >= 3 or k == 2:
        ax.set_xlabel("Predicted 2020", fontsize=7)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[0, :], shrink=0.8, pad=0.01, aspect=30)
cb.set_label("Share of observed area (%)", fontsize=7)
cb.ax.tick_params(labelsize=6.5, length=2.5, direction="in")
cb.ax.yaxis.set_minor_locator(MultipleLocator(10))

ax = axs[1, 2]
x = np.arange(K)
wbar = 0.16
for k, (name, _s, c, _m) in enumerate(MODELS):
    ax.bar(x + (k - 2) * wbar, f1[name], wbar, color=c, label=name, edgecolor="white", lw=0.2)
ax.set_xticks(x, SHORT, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.3)
ax.set_ylim(0.4, 1.12)
ax.set_yticks(np.round(np.arange(0.4, 1.01, 0.1), 1))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(axis="y", labelsize=6.5, direction="in", which="both", right=True)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", lw=0.4, color="0.88")
ax.set_axisbelow(True)
ax.set_ylabel("F1 score", fontsize=7)
ax.set_title("(f) F1 per class group", fontsize=7.5, loc="left", pad=3)
leg = ax.legend(fontsize=5.8, ncol=2, frameon=False, loc="upper right", handlelength=1.0, columnspacing=0.8,
                borderaxespad=0.3, handletextpad=0.4)

fig.canvas.draw()
rnd = fig.canvas.get_renderer()
lb = leg.get_window_extent(rnd)
top_bar = max(pt.get_window_extent(rnd).y1 for pt in ax.patches)
bars_in_legend = [pt for pt in ax.patches if pt.get_window_extent(rnd).overlaps(lb)]
print("bars under legend:", len(bars_in_legend))
assert not bars_in_legend and top_bar < lb.y0, "legend covers bars"
assert ax.bbox.contains(lb.x0, lb.y0) and ax.bbox.contains(lb.x1, lb.y1), "legend outside the axes"
for name, _s, _c, _m in MODELS:
    print(f"{name:14s} diag %: " + " ".join(f"{v:5.1f}" for v in np.diag(cms[name])))
fig.savefig(OUT / "fig09_confusion.pdf", dpi=600)
fig.savefig(OUT / "fig09_confusion.png", dpi=600)
