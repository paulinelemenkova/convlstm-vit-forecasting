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

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Helvetica"], "font.size": 8,
                     "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                     "ytick.right": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans", "Helvetica"])).lower()

GROUPS = {1: ("Natural forest", [10, 11, 12, 20], "#1b6e2d"), 2: ("Plantation", [14], "#8fce5a"),
          3: ("Mangrove", [15], "#b24fd1"), 4: ("Woody crops", [4], "#d9b25f"),
          5: ("Other agriculture", [3, 5, 6], "#3a86a7"), 6: ("Built-up", [1, 2], "#c0392b"),
          7: ("Other land", [7, 8, 9, 16, 18, 19], "#a3a3a3")}
NAMES = [v[0] for v in GROUPS.values()]
COLS = [v[2] for v in GROUPS.values()]
LUT = np.zeros(256, dtype=np.uint8)
for gid, (_n, codes, _c) in GROUPS.items():
    LUT[codes] = gid
YEARS = list(range(1990, 2021))

with rasterio.open(find_file("VLUCD_L2_250m_2020.tif", IN)) as src:
    tr, crs, shape = src.transform, src.crs, src.shape
valid = rasterize(((g, 1) for g in gpd.read_file(ADM0).to_crs(crs).geometry), out_shape=shape, transform=tr,
                  dtype="uint8") == 1
maps: dict[str, np.ndarray] = {}
for y in YEARS:
    with rasterio.open(find_file(f"VLUCD_L2_250m_{y}.tif", IN)) as src:
        maps[str(y)] = LUT[src.read(1)]
        valid &= maps[str(y)] > 0
for m, lab in (("convlstm", "ConvLSTM 2030"), ("vit", "ViT 2030")):
    with rasterio.open(PRED / f"{m}_2030.tif") as src:
        maps[lab] = src.read(1)
        valid &= maps[lab] > 0
r = 6371.0072
lat = tr.f + np.arange(shape[0]) * tr.e
row_area = r**2 * np.radians(abs(tr.a)) * np.abs(np.sin(np.radians(lat)) - np.sin(np.radians(lat + tr.e)))
w = np.broadcast_to(row_area[:, None], shape)[valid]
area = pd.DataFrame({k: np.bincount(v[valid], weights=w, minlength=8)[1:] / 1000 for k, v in maps.items()},
                    index=NAMES).T
area["Forest"] = area[["Natural forest", "Plantation", "Mangrove"]].sum(axis=1)
area.to_csv(OUT / "fig10_trajectory_2030_data.csv", float_format="%.3f")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained", width_ratios=[1.35, 1])
fig.get_layout_engine().set(w_pad=0.02, wspace=0.04)

obs = area.loc[[str(y) for y in YEARS], NAMES]
a1.stackplot(YEARS, obs.T.values, colors=COLS, lw=0.3, edgecolor="white")
xb = {"ConvLSTM 2030": 2025.5, "ViT 2030": 2030.5}
for lab, x in xb.items():
    bottom = 0.0
    for n, c in zip(NAMES, COLS, strict=True):
        a1.bar(x, area.loc[lab, n], 3.6, bottom=bottom, color=c, edgecolor="white", lw=0.3)
        bottom += area.loc[lab, n]
a1.axvline(2022.6, color="0.25", lw=0.8, ls="--")
a1.text(2021.9, 352, "observed", ha="right", va="top", fontsize=6.8, color="0.25")
a1.text(2023.3, 352, "forecast", ha="left", va="top", fontsize=6.8, color="0.25")
a1.set_xlim(1989.5, 2033)
a1.set_ylim(0, 365)
a1.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020, *xb.values()],
              ["1990", "1995", "2000", "2005", "2010", "2015", "2020", "ConvLSTM\n2030", "ViT\n2030"])
for t in a1.get_xticklabels()[-2:]:
    t.set_fontsize(6.3)
a1.xaxis.set_minor_locator(MultipleLocator(1))
a1.yaxis.set_major_locator(MultipleLocator(50))
a1.yaxis.set_minor_locator(AutoMinorLocator())
a1.tick_params(which="major", length=4)
a1.tick_params(which="minor", length=2)
a1.set_xlabel("Year")
a1.set_ylabel("Area (10$^3$ km$^2$)")
a1.text(0.02, 0.98, "(a)", transform=a1.transAxes, ha="left", va="top", weight="bold", fontsize=10)
mid = obs.loc["2005"].cumsum() - obs.loc["2005"] / 2
for n in NAMES:
    if obs.loc["2005", n] > 12:
        a1.text(2005, mid[n], n, ha="center", va="center", fontsize=6.8,
                color="white" if n in ("Natural forest", "Other agriculture") else "black")

periods = [("1990", "2000", "1990–2000"), ("2000", "2010", "2000–2010"), ("2010", "2020", "2010–2020"),
           ("2020", "ConvLSTM 2030", "2020–2030\nConvLSTM"), ("2020", "ViT 2030", "2020–2030\nViT")]
series = [("Natural forest", "#1b6e2d"), ("Plantation", "#8fce5a"), ("Forest", "#555555")]
x = np.arange(len(periods))
bw = 0.26
rows = []
for k, (n, c) in enumerate(series):
    vals = [100 * (area.loc[b, n] - area.loc[a, n]) / area.loc[a, n] for a, b, _l in periods]
    bars = a2.bar(x + (k - 1) * bw, vals, bw, color=c, edgecolor="white", lw=0.3,
                  label="All forest" if n == "Forest" else n)
    for b_ in list(bars)[3:]:
        b_.set_hatch("////")
        b_.set_edgecolor("white")
    rows += [{"series": n, "period": lab.replace("\n", " "), "change_pct": v} for (_a, _b, lab), v in
             zip(periods, vals, strict=True)]
pd.DataFrame(rows).to_csv(OUT / "fig10_change_per_decade.csv", index=False, float_format="%.2f")
a2.axhline(0, color="0.3", lw=0.7)
a2.axvline(2.5, color="0.25", lw=0.8, ls="--")
a2.set_xticks(x, [p[2] for p in periods], fontsize=6.6)
a2.tick_params(axis="x", length=0)
a2.yaxis.set_major_locator(MultipleLocator(10))
a2.yaxis.set_minor_locator(AutoMinorLocator(2))
a2.set_ylabel("Net change per decade (%)")
a2.grid(axis="y", lw=0.4, color="0.88")
a2.set_axisbelow(True)
lo = min(r_["change_pct"] for r_ in rows)
hi = max(r_["change_pct"] for r_ in rows)
a2.set_ylim(np.floor(lo / 10) * 10 - 2, np.ceil(hi / 10) * 10 + 8)
a2.text(0.03, 0.98, "(b)", transform=a2.transAxes, ha="left", va="top", weight="bold", fontsize=10)
leg = a2.legend(loc="upper right", fontsize=6.6, ncol=1, handlelength=1.2, borderaxespad=0.3)

fig.canvas.draw()
rnd = fig.canvas.get_renderer()
lb = leg.get_window_extent(rnd)
hit = [p for p in a2.patches if p.get_window_extent(rnd).overlaps(lb)]
assert not hit, f"legend covers {len(hit)} bars"
print(area.loc[["1990", "2000", "2010", "2020", "ConvLSTM 2030", "ViT 2030"]].round(1).to_string())
print(pd.DataFrame(rows).pivot(index="period", columns="series", values="change_pct").round(1).to_string())
fig.savefig(OUT / "fig10_trajectory_2030.pdf", dpi=600)
fig.savefig(OUT / "fig10_trajectory_2030.png", dpi=600)
