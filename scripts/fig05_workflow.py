"""Workflow diagram for Paper A (Figure 5): JAXA land cover maps, change analysis, forecasting models.

Five rows (data, pre-processing, analysis and modelling, validation, outputs) on a three-column grid.
Box heights are derived from the wrapped text; the script verifies that every text block
fits its box, that every arrow starts and ends on a box edge, and that no arrow crosses text.
Coordinates are in inches (1 data unit = 72 pt). Environment variable: FIG_OUT.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

OUT = Path(os.environ.get("FIG_OUT", "./figs"))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
assert (
    "dejavu"
    not in findfont(FontProperties(family=plt.rcParams["font.sans-serif"])).lower()
)

W = 5.45  # in, MDPI text width (13.9 cm)
FS_ROW, FS_HEAD, FS_BODY, FS_TOOL = 9.5, 9.0, 8.5, 8.0
LH = 1.2  # line spacing factor
PAD = 0.075  # in, inner padding (top/bottom)
GAP_X = 0.10  # in, gap between columns
GAP_ROW = 0.34  # in, vertical gap between rows (row title + arrows)
NCOL = 3
COLW = (W - (NCOL - 1) * GAP_X) / NCOL
INNER_GAP = 3.0  # pt, gap between head, body and tool lines

STAGE = {  # light fill, dark outline of the same Okabe-Ito hue
    "data": ("#DCEBF7", "#0072B2"),
    "proc": ("#FBE7C6", "#A0700A"),
    "anal": ("#D5EFE6", "#007A5A"),
    "val": ("#E8E4F5", "#6A51A3"),
    "out": ("#F3DDEB", "#A8527F"),
}
ROW_TITLES = [
    "1  Data",
    "2  Pre-processing",
    "3  Analysis",
    "4  Validation",
    "5  Outputs",
]
ROW_STAGE = ["data", "proc", "anal", "val", "out"]


def vis_len(s: str) -> int:
    """Visible length of a string, ignoring mathtext markup."""
    return len(re.sub(r"\$|\\mathit\{|\^\{|\}", "", s))


def wrap_vis(text: str, width: int) -> list[str]:
    """Greedy word wrap using the visible (rendered) length of each word."""
    lines: list[str] = []
    cur = ""
    for w in text.split(" "):
        cand = f"{cur} {w}" if cur else w
        if cur and vis_len(cand) > width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


@dataclass
class Box:
    key: str
    row: int
    c0: int  # first column (0-based)
    c1: int  # last column (inclusive)
    head: str
    body: str
    tool: str = ""
    x0: float = 0.0
    x1: float = 0.0
    y0: float = 0.0
    y1: float = 0.0
    texts: list[Text] = field(default_factory=list)

    @property
    def wrap(self) -> int:
        """Characters per line from the box width (8.5-pt Nimbus Sans ~ 0.058 in / char)."""
        width = (self.c1 - self.c0 + 1) * COLW + (self.c1 - self.c0) * GAP_X - 0.16
        return int(width / 0.061)

    def lines(self) -> tuple[list[str], list[str], list[str]]:
        return (
            wrap_vis(self.head, int(self.wrap * 0.9)),
            wrap_vis(self.body, self.wrap),
            wrap_vis(self.tool, self.wrap) if self.tool else [],
        )

    def content_h(self) -> float:
        h, b, t = self.lines()
        pt = (len(h) * FS_HEAD + len(b) * FS_BODY + len(t) * FS_TOOL) * LH
        pt += INNER_GAP * ((1 if b else 0) + (1 if t else 0))
        return pt / 72.0


BOXES = [
    Box(
        "jaxa",
        0,
        0,
        1,
        "JAXA HRLULC maps of Vietnam, v21.09",
        "31 annual maps 1990–2020, 250 m; Level 1 (10 classes) and Level 2 (18 classes)",
    ),
    Box(
        "ref",
        0,
        2,
        2,
        "Reference data",
        "Province boundaries (HDX, 34 provinces); 30 m class areas (Phan et al., 2021)",
    ),
    Box(
        "prep",
        1,
        0,
        2,
        "Clipping, grouping and encoding",
        "Maps clipped to the national boundary; cell area from latitude; 18 Level-2 classes "
        "grouped into 7; one-hot encoding; 64 × 64-cell patches, 20% of spatial blocks held out",
        "Python: rasterio, GeoPandas, NumPy",
    ),
    Box(
        "change",
        2,
        0,
        0,
        "Land cover change",
        "Class areas 1990–2020; transition matrices per decade; forest loss and gain; bias against 30 m areas",
        "Python: pandas",
    ),
    Box(
        "dl",
        2,
        1,
        1,
        "Deep learning models",
        "ConvLSTM and Temporal ViT; inputs years t−4 to t, target t+10 (t\u00a0=\u00a01994–2000); class-weighted loss; one\u00a0seed per model",
        "Python: PyTorch",
    ),
    Box(
        "base",
        2,
        2,
        2,
        "Baselines",
        "Persistence; CA–Markov with 2000–2010 transitions; Random Forest with 3 × 3 neighbourhood",
        "Python: scikit-learn",
    ),
    Box(
        "hind",
        3,
        1,
        2,
        "Hindcast of 2020",
        "Prediction from 2006–2010 against the observed 2020 map: overall accuracy, kappa, "
        "quantity and allocation disagreement, F1, figure of merit",
    ),
    Box(
        "o_dyn",
        4,
        0,
        0,
        "Forest dynamics",
        "Areas, transitions and change maps, 1990–2020",
    ),
    Box(
        "o_cmp",
        4,
        1,
        1,
        "Model comparison",
        "Accuracy table, confusion matrices and attention rollout",
    ),
    Box(
        "o_fc",
        4,
        2,
        2,
        "Forecast for 2030",
        "Class maps from 2016–2020 inputs; agreement of the two models",
    ),
]
BOX = {b.key: b for b in BOXES}
EDGES = [
    ("jaxa", "prep"),
    ("ref", "prep"),
    ("prep", "change"),
    ("prep", "dl"),
    ("prep", "base"),
    ("dl", "hind"),
    ("base", "hind"),
    ("change", "o_dyn"),
    ("hind", "o_cmp"),
    ("hind", "o_fc"),
]


def layout() -> float:
    y = 0.0
    for r in range(len(ROW_TITLES)):
        y -= GAP_ROW if r > 0 else 0.22  # first row: space for its title only
        row = [b for b in BOXES if b.row == r]
        h = max(b.content_h() for b in row) + 2 * PAD
        for b in row:
            b.x0 = b.c0 * (COLW + GAP_X)
            b.x1 = b.c1 * (COLW + GAP_X) + COLW
            b.y1, b.y0 = y, y - h
        y -= h
    return -y


H = layout()
fig = plt.figure(figsize=(W, H))
ax: Axes = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, W)
ax.set_ylim(-H, 0)
ax.axis("off")

# ---------------------------------------------------------------- boxes and row titles
row_titles: list[Text] = []
for r, title in enumerate(ROW_TITLES):
    top = max(b.y1 for b in BOXES if b.row == r)
    row_titles.append(
        ax.text(
            0.0,
            top + 0.035,
            title,
            ha="left",
            va="bottom",
            fontsize=FS_ROW,
            weight="bold",
            color=STAGE[ROW_STAGE[r]][1],
        )
    )

for b in BOXES:
    fill, edge = STAGE[ROW_STAGE[b.row]]
    ax.add_patch(
        FancyBboxPatch(
            (b.x0, b.y0),
            b.x1 - b.x0,
            b.y1 - b.y0,
            boxstyle="round,pad=0,rounding_size=0.05",
            fc=fill,
            ec=edge,
            lw=0.9,
            zorder=2,
        )
    )
    h, bd, tl = b.lines()
    cx = (b.x0 + b.x1) / 2
    y = (b.y0 + b.y1) / 2 + b.content_h() / 2  # centre the whole text block vertically
    for lines_, fs, kw in (
        (h, FS_HEAD, {"weight": "bold"}),
        (bd, FS_BODY, {}),
        (tl, FS_TOOL, {"style": "italic", "color": "0.25"}),
    ):
        if not lines_:
            continue
        b.texts.append(
            ax.text(
                cx,
                y,
                "\n".join(lines_),
                ha="center",
                va="top",
                fontsize=fs,
                linespacing=LH,
                zorder=4,
                **kw,
            )
        )
        y -= (len(lines_) * fs * LH + INNER_GAP) / 72.0

# ---------------------------------------------------------------- arrows (vertical only)
ARROW_KW = dict(
    arrowstyle="-|>",
    mutation_scale=12,
    lw=0.9,
    color="0.25",
    shrinkA=0,
    shrinkB=0,
    zorder=3,
)
segments: list[tuple[str, str, float, float, float]] = []  # a, b, x, y_start, y_end
for a_key, b_key in EDGES:
    a, b = BOX[a_key], BOX[b_key]
    lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
    # arrows sit in the right part of the shared span, clear of the left-aligned row titles
    x = hi - 0.42 if (hi - lo) < 1.2 * COLW else (b.x0 + b.x1) / 2
    ax.add_patch(FancyArrowPatch((x, a.y0), (x, b.y1), **ARROW_KW))
    segments.append((a_key, b_key, x, a.y0, b.y1))

# ---------------------------------------------------------------- verification (pixel space)
fig.canvas.draw()
rnd = fig.canvas.get_renderer()  # type: ignore[attr-defined]
fails = 0
for b in BOXES:
    (bx0, by0), (bx1, by1) = ax.transData.transform([(b.x0, b.y0), (b.x1, b.y1)])
    for t in b.texts:
        e = t.get_window_extent(rnd)
        if not (
            e.x0 >= bx0 + 2 and e.x1 <= bx1 - 2 and e.y0 >= by0 + 2 and e.y1 <= by1 - 2
        ):
            fails += 1
            print("TEXT NOT INSIDE BOX:", b.key, repr(t.get_text()[:40]))
for a_key, b_key, x, ys, ye in segments:
    a, b = BOX[a_key], BOX[b_key]
    ok_x = a.x0 < x < a.x1 and b.x0 < x < b.x1
    ok_y = np.isclose(ys, a.y0) and np.isclose(ye, b.y1)
    if not (ok_x and ok_y):
        fails += 1
        print("ARROW NOT CONNECTED:", a_key, "->", b_key)
    # arrow must not pass through any text (row titles or box texts)
    p0 = ax.transData.transform((x, ys))
    p1 = ax.transData.transform((x, ye))
    for t in row_titles + [t for bb in BOXES for t in bb.texts]:
        e = t.get_window_extent(rnd).expanded(1.0, 1.0)
        if (
            e.x0 - 2 <= p0[0] <= e.x1 + 2
            and min(p0[1], p1[1]) < e.y1
            and max(p0[1], p1[1]) > e.y0
        ):
            fails += 1
            print("ARROW CROSSES TEXT:", a_key, "->", b_key, repr(t.get_text()[:30]))
print("verification failures:", fails)
fig.savefig(OUT / "fig05_workflow.pdf")
fig.savefig(OUT / "fig05_workflow.png", dpi=600)
print(f"size: {W:.2f} x {H:.2f} in")
