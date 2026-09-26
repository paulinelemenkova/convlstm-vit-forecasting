from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.legend import Legend
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

IN = Path(os.environ.get("FIG_IN", "."))
OUT = Path(os.environ.get("FIG_OUT", "./figs"))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans", "Helvetica"])).lower()

GROUPS: dict[str, tuple[list[int], str, str]] = {
    "Natural forest": ([10, 11, 12, 20], "#1b6e2d", "o"),
    "Plantation": ([14], "#8fce5a", "s"),
    "Mangrove": ([15], "#b24fd1", "D"),
    "Woody crops": ([4], "#d9b25f", "^"),
    "Other agriculture": ([3, 5, 6], "#3a86a7", "v"),
    "Built-up": ([1, 2], "#c0392b", "P"),
    "Other land": ([7, 8, 9, 16, 18, 19], "#a3a3a3", "X"),
}

d = pd.read_csv(IN / "class_areas_km2.csv")
d = d[d.level == "L2"]
wide = d.pivot(index="year", columns="code", values="area_km2")
area = pd.DataFrame({g: wide[codes].sum(axis=1) for g, (codes, _, _) in GROUPS.items()}) / 1000.0
area.to_csv(OUT / "fig06_area_timeseries_data.csv", float_format="%.3f")
years = area.index.values


def style(ax: Axes) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major", length=4, width=0.8)
    ax.tick_params(which="minor", length=2, width=0.6)
    ax.grid(which="major", lw=0.5, color="0.88")
    ax.set_axisbelow(True)
    ax.set_xlim(1989.5, 2020.5)


def covers_data(leg: Legend, ax: Axes) -> int:
    fig = ax.figure
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    n = 0
    for ln in ax.get_lines():
        p = ax.transData.transform(np.column_stack(ln.get_data()))
        n += int(((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) & (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).sum())
    return n


fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 3.2), layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0)

cols = [spec[1] for spec in GROUPS.values()]
a1.stackplot(years, area.T.values, colors=cols, linewidth=0.3, edgecolor="white")
style(a1)
a1.set_ylim(0, 365)
a1.yaxis.set_major_locator(MultipleLocator(50))
a1.set_xlabel("Year")
a1.set_ylabel("Area (10$^3$ km$^2$)")
a1.text(0.02, 0.98, "(a)", transform=a1.transAxes, ha="left", va="top", weight="bold", fontsize=10)
mid = area.loc[2005].cumsum() - area.loc[2005] / 2
for g in GROUPS:
    h = area.loc[2005, g]
    if h > 12:
        a1.text(2005, mid[g], g, ha="center", va="center", fontsize=7.5,
                color="white" if g in ("Natural forest", "Other agriculture") else "black")

ch = area - area.loc[1990]
for g, (_codes, c, m) in GROUPS.items():
    a2.plot(years, ch[g], color=c, marker=m, ms=3.2, lw=1.2, mec="white", mew=0.3, label=g, markevery=2)
a2.axhline(0, color="0.3", lw=0.7)
style(a2)
lo, hi = np.floor(ch.values.min() / 5) * 5 - 2, np.ceil(ch.values.max() / 5) * 5 + 2
a2.set_ylim(lo, hi)
a2.yaxis.set_major_locator(MultipleLocator(5))
a2.set_xlabel("Year")
a2.set_ylabel("Change since 1990 (10$^3$ km$^2$)")
a2.text(0.02, 0.98, "(b)", transform=a2.transAxes, ha="left", va="top", weight="bold", fontsize=10)
h, lab = a2.get_legend_handles_labels()
leg = fig.legend(h, lab, loc="outside lower center", ncol=4, handlelength=1.8, columnspacing=0.9,
                 borderaxespad=0.0, borderpad=0.1)

hits = covers_data(leg, a2)
print("legend covers data points:", hits)
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_lb = leg.get_window_extent(_r)
print("legend x-range within panels:", _lb.x0 >= a1.get_tightbbox(_r).x0 - 1 and _lb.x1 <= a2.get_position().x1 * fig.bbox.width + 1)
print(area.loc[[1990, 2000, 2010, 2020]].round(1).to_string())
print(ch.loc[[2020]].round(1).to_string())
fig.savefig(OUT / "fig06_area_timeseries.pdf")
fig.savefig(OUT / "fig06_area_timeseries.png", dpi=600)
