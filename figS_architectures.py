"""Supplementary Figures S1 and S2: block diagrams of the ConvLSTM and the Temporal ViT (Paper A).

Values follow scripts/04_train_dl.py: 5 input years, K = 7 classes, 64 x 64-cell tiles,
ConvLSTM 2 layers x 64 channels (3 x 3), ViT 8 x 8-cell tokens, d = 128, 4 heads, 4 layers, MLP 512.
Coordinates in inches. Environment variable: FIG_OUT.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.cal": "Nimbus Sans",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans"])).lower()

C = {  # light fill, dark outline (Okabe-Ito hues, as in Figure 5)
    "in": ("#DCEBF7", "#0072B2"),
    "core": ("#D5EFE6", "#007A5A"),
    "op": ("#FBE7C6", "#A0700A"),
    "out": ("#F3DDEB", "#A8527F"),
    "att": ("#E8E4F5", "#6A51A3"),
    "grey": ("#F2F2F2", "#666666"),
}
TEXTS: list[tuple[Text, tuple[float, float, float, float]]] = []


def box(ax: Axes, x: float, y: float, w: float, h: float, txt: str, kind: str,
        fs: float = 8.0, bold_first: bool = True, lw: float = 0.9) -> tuple[float, float, float, float]:
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=fill, ec=edge, lw=lw, zorder=2))
    lines = txt.split("\n")
    if bold_first and len(lines) > 1:
        t1 = ax.text(x + w / 2, y + h / 2 + (len(lines) - 1) * fs * 0.6 / 72, lines[0], ha="center",
                     va="center", fontsize=fs, weight="bold", zorder=4)
        t2 = ax.text(x + w / 2, y + h / 2 - fs * 0.6 / 72, "\n".join(lines[1:]), ha="center",
                     va="center", fontsize=fs - 0.5, linespacing=1.15, zorder=4)
        TEXTS.extend([(t1, (x, y, w, h)), (t2, (x, y, w, h))])
    else:
        t = ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                    weight="bold" if bold_first else "normal", linespacing=1.15, zorder=4)
        TEXTS.append((t, (x, y, w, h)))
    return x, y, w, h


def arrow(ax: Axes, p0: tuple[float, float], p1: tuple[float, float], color: str = "0.25",
          ls: str = "-", cs: str = "arc3") -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=0.9, color=color,
                                 shrinkA=0, shrinkB=0, zorder=3, linestyle=ls, connectionstyle=cs))


def canvas(w: float, h: float) -> tuple[plt.Figure, Axes]:
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def check(fig: plt.Figure, ax: Axes, name: str) -> None:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    bad = 0
    for t, (x, y, w, h) in TEXTS:
        e = t.get_window_extent(r)
        (bx0, by0), (bx1, by1) = ax.transData.transform([(x, y), (x + w, y + h)])
        if not (e.x0 >= bx0 + 1 and e.x1 <= bx1 - 1 and e.y0 >= by0 + 1 and e.y1 <= by1 - 1):
            bad += 1
            print(name, "TEXT OUTSIDE BOX:", repr(t.get_text()[:40]))
    print(name, "verification failures:", bad)
    TEXTS.clear()


# ================================================================ Figure S1: ConvLSTM
W, H = 7.0, 5.45
fig, ax = canvas(W, H)
years = ["2006", "2007", "2008", "2009", "2010"]
x0, dx, bw = 0.66, 1.08, 0.84
ax.text(0.0, H - 0.12, "(a) ConvLSTM unrolled over the five input years (2020 hindcast)", fontsize=9,
        weight="bold", va="top")
y_in, y_l1, y_l2 = 2.85, 3.70, 4.55
bh = 0.58
ax.text(0.0, y_in + bh / 2, "Input", fontsize=8, va="center", style="italic")
ax.text(0.0, y_l1 + bh / 2, "Layer 1", fontsize=8, va="center", style="italic")
ax.text(0.0, y_l2 + bh / 2, "Layer 2", fontsize=8, va="center", style="italic")
for i, yr in enumerate(years):
    x = x0 + i * dx
    box(ax, x, y_in, bw, bh, f"$\\mathit{{X}}_{{{yr}}}$\n7 × 64 × 64", "in", fs=8)
    box(ax, x, y_l1, bw, bh, "ConvLSTM\n64 ch, 3 × 3", "core", fs=8)
    box(ax, x, y_l2, bw, bh, "ConvLSTM\n64 ch, 3 × 3", "core", fs=8)
    arrow(ax, (x + bw / 2, y_in + bh), (x + bw / 2, y_l1))
    arrow(ax, (x + bw / 2, y_l1 + bh), (x + bw / 2, y_l2))
    if i > 0:
        xp = x0 + (i - 1) * dx + bw
        arrow(ax, (xp, y_l1 + bh / 2), (x, y_l1 + bh / 2))
        arrow(ax, (xp, y_l2 + bh / 2), (x, y_l2 + bh / 2))
        ax.text((xp + x) / 2, y_l2 + bh / 2 + 0.05, "$\\mathit{H},\\mathit{C}$", fontsize=7,
                ha="center", va="bottom", color="0.3")
        ax.text((xp + x) / 2, y_l1 + bh / 2 + 0.05, "$\\mathit{H},\\mathit{C}$", fontsize=7,
                ha="center", va="bottom", color="0.3")
ax.text(x0 - 0.08, y_l1 + bh / 2 + 0.05, "0", fontsize=7, ha="right", va="bottom", color="0.3")
ax.text(x0 - 0.08, y_l2 + bh / 2 + 0.05, "0", fontsize=7, ha="right", va="bottom", color="0.3")
xr = x0 + 4 * dx + bw + 0.26
box(ax, xr, y_l2, 0.88, bh, "1 × 1 conv\n64 → 7", "op", fs=8)
arrow(ax, (x0 + 4 * dx + bw, y_l2 + bh / 2), (xr, y_l2 + bh / 2))
box(ax, xr, y_l1, 0.88, bh, "Softmax\n7 × 64 × 64", "op", fs=8)
arrow(ax, (xr + 0.44, y_l2), (xr + 0.44, y_l1 + bh))
box(ax, xr, y_in, 0.88, bh, "Class map\n2020, 64 × 64", "out", fs=8)
arrow(ax, (xr + 0.44, y_l1), (xr + 0.44, y_in + bh))
ax.text(xr + 0.44, y_in - 0.08, "argmax; tiles merged\nwith 50% overlap", fontsize=7, ha="center",
        va="top", color="0.3")
ax.text(x0, y_in - 0.12, "One-hot maps of the seven class groups; 459,463 trainable parameters; "
        "class-weighted cross-entropy; AdamW", fontsize=7.5, va="top", color="0.25")

# (b) inside one cell
ax.text(0.0, 2.12, "(b) One ConvLSTM cell at year t (Equation 1)", fontsize=9, weight="bold", va="top")
yb, hb = 0.55, 0.62
box(ax, 0.05, yb + 0.55, 1.15, hb, "Concatenate\n$[\\mathit{X}_t,\\ \\mathit{H}_{t-1}]$", "in", fs=8)
box(ax, 1.45, yb + 0.55, 1.15, hb, "Conv 3 × 3\n→ 4 × 64 ch", "op", fs=8)
arrow(ax, (1.20, yb + 0.55 + hb / 2), (1.45, yb + 0.55 + hb / 2))
gates = [("$\\mathbf{i}_t$", "σ", "input gate"), ("$\\mathbf{f}_t$", "σ", "forget gate"),
         ("$\\mathbf{o}_t$", "σ", "output gate"), ("$\\mathbf{g}_t$", "tanh", "candidate")]
gx = 2.90
for k, (g, f, lab) in enumerate(gates):
    yy = yb + 1.30 - k * 0.42
    box(ax, gx, yy, 1.05, 0.34, f"{g} = {f}(·)", "core", fs=7.5, bold_first=False)
    ax.text(gx + 1.12, yy + 0.17, lab, fontsize=7, va="center", color="0.3")
    arrow(ax, (2.60, yb + 0.55 + hb / 2), (gx, yy + 0.17), cs="arc3,rad=0")
box(ax, 5.05, yb + 0.85, 1.90, 0.62,
    "Cell state\n$\\mathit{C}_t=\\mathbf{f}_t\\odot\\mathit{C}_{t-1}+\\mathbf{i}_t\\odot\\mathbf{g}_t$", "att",
    fs=8)
box(ax, 5.05, yb - 0.05, 1.90, 0.62,
    "Hidden state\n$\\mathit{H}_t=\\mathbf{o}_t\\odot\\tanh(\\mathit{C}_t)$", "att", fs=8)
arrow(ax, (6.0, yb + 0.85), (6.0, yb + 0.57))
arrow(ax, (4.72, yb + 1.30 - 0.42 + 0.17), (5.05, yb + 0.85 + 0.31), color="0.45")
ax.text(4.84, yb + 1.20, "i, f, g", fontsize=7, color="0.3", ha="center", style="italic")
arrow(ax, (4.72, yb + 1.30 - 2 * 0.42 + 0.17), (5.05, yb - 0.05 + 0.31), color="0.45")
ax.text(4.84, yb + 0.62, "o", fontsize=7, color="0.3", ha="center", style="italic")
ax.text(6.06, yb + 1.58, "$\\mathit{C}_{t-1}$ from year t − 1", fontsize=7, color="0.3", ha="left", va="center")
arrow(ax, (6.0, yb + 1.72), (6.0, yb + 1.47), color="0.45")
ax.text(5.05, yb - 0.17, "$\\mathit{H}_t,\\ \\mathit{C}_t$ to year t + 1 and to layer 2", fontsize=7,
        color="0.3", va="top")
check(fig, ax, "S1")
fig.savefig(OUT / "figS1_convlstm_architecture.pdf")
fig.savefig(OUT / "figS1_convlstm_architecture.png", dpi=600)
plt.close(fig)

# ================================================================ Figure S2: Temporal ViT
W, H = 6.15, 8.3
fig, ax = canvas(W, H)
cx, bw2, bh2 = 1.80, 3.4, 0.56
xm = cx - bw2 / 2
steps = [
    ("Input sequence\n5 years × 7 one-hot layers × 64 × 64 cells (2006–2010)", "in"),
    ("Tokenization\n8 × 8-cell tokens (2 km): 5 × 64 = 320 tokens of 7 × 8 × 8 = 448 values", "op"),
    ("Linear embedding\n448 → d = 128", "op"),
    ("Add position embeddings\nlearned spatial (64 × 128) + temporal (5 × 128)", "op"),
]
y = H - 0.30 - bh2
pos: dict[str, tuple[float, float]] = {}
for k, (t, kind) in enumerate(steps):
    box(ax, xm, y, bw2, bh2, t, kind, fs=8)
    if k:
        arrow(ax, (cx, y + bh2 + 0.22), (cx, y + bh2))
    pos[f"s{k}"] = (y, y + bh2)
    y -= bh2 + 0.22
# encoder block (repeated 4x)
enc_top = y + bh2 + 0.22 - 0.02
sub = [
    ("LayerNorm", "grey"),
    ("Multi-head self-attention\n4 heads, $d_k$ = 32 (Equation 2)", "att"),
    ("Residual addition", "grey"),
    ("LayerNorm", "grey"),
    ("MLP\n128 → 512 → 128, GELU", "core"),
    ("Residual addition", "grey"),
]
sh = [0.30, 0.52, 0.30, 0.30, 0.52, 0.30]
yy = enc_top - 0.30
ys: list[tuple[float, float]] = []
for (t, kind), hgt in zip(sub, sh, strict=True):
    yy -= hgt
    box(ax, xm + 0.35, yy, bw2 - 0.70, hgt, t, kind, fs=7.5, bold_first="\n" in t)
    ys.append((yy, yy + hgt))
    yy -= 0.14
enc_bot = yy - 0.02
ax.add_patch(FancyBboxPatch((xm, enc_bot), bw2, enc_top - enc_bot, boxstyle="round,pad=0,rounding_size=0.06",
                            fc="none", ec="#007A5A", lw=1.1, ls="--", zorder=1))
ax.text(xm + 0.08, enc_top - 0.06, "Transformer encoder layer (pre-norm) × 4", fontsize=8, weight="bold",
        color="#007A5A", va="top")
arrow(ax, (cx, pos["s3"][0]), (cx, enc_top))
for k in range(1, len(ys)):
    arrow(ax, (cx, ys[k - 1][0]), (cx, ys[k][1]))
# residual skips (left side): from the stream above each LayerNorm to the residual addition
for a, b in ((0, 2), (3, 5)):
    ytop = ys[a][1] + 0.07
    ybot = (ys[b][0] + ys[b][1]) / 2
    xl = xm + 0.17
    ax.plot([cx, xl, xl], [ytop, ytop, ybot], color="0.45", lw=0.8, zorder=3, solid_joinstyle="miter")
    arrow(ax, (xl, ybot), (xm + 0.35, ybot), color="0.45")
y = enc_bot - 0.22 - bh2
arrow(ax, (cx, enc_bot), (cx, y + bh2))
tail = [
    ("LayerNorm and temporal mean\n320 tokens → 64 tokens × 128 (mean over the 5 years)", "op"),
    ("Linear head\n128 → 7 × 8 × 8 per token; reshape to 7 × 64 × 64", "op"),
    ("Softmax and class map 2020\n7 class probabilities per 250 m cell → argmax", "out"),
]
for k, (t, kind) in enumerate(tail):
    box(ax, xm, y, bw2, bh2, t, kind, fs=8)
    if k:
        arrow(ax, (cx, y + bh2 + 0.22), (cx, y + bh2))
    y -= bh2 + 0.22
ax.text(xm, y + bh2 + 0.12, "917,440 trainable parameters; class-weighted cross-entropy; AdamW",
        fontsize=7.5, va="top", color="0.25")
y_min = y + bh2 - 0.15
# attention rollout branch (right side)
xa, wa = 3.85, 2.2
att_y = ys[1]
ra = [
    ("Attention matrices\n$\\mathbf{A}_l$ of the 4 layers,\nheads averaged", "att"),
    ("Residual correction\n$\\tilde{\\mathbf{A}}_l = 0.5\\,\\mathbf{A}_l + 0.5\\,\\mathbf{I}$,\nrows normalized", "att"),
    ("Rollout\n$\\mathbf{R} = \\tilde{\\mathbf{A}}_4\\tilde{\\mathbf{A}}_3"
     "\\tilde{\\mathbf{A}}_2\\tilde{\\mathbf{A}}_1$", "att"),
    ("Attention per token\nand input year\n(Figure 12)", "out"),
]
ya = att_y[1] + 0.30
for k, (t, kind) in enumerate(ra):
    hgt = 0.74 if t.count("\n") >= 2 else 0.58
    box(ax, xa, ya - hgt, wa, hgt, t, kind, fs=7.5)
    if k == 0:
        arrow(ax, (xm + bw2 - 0.35, (att_y[0] + att_y[1]) / 2), (xa, ya - hgt / 2), color="#6A51A3", ls="--")
    else:
        arrow(ax, (xa + wa / 2, prev), (xa + wa / 2, ya))
    prev = ya - hgt
    ya = prev - 0.22
ax.text(xa, att_y[1] + 0.42, "Attention rollout (interpretation)", fontsize=8, weight="bold", color="#6A51A3")
ax.set_ylim(y_min, H)
fig.set_size_inches(W, H - y_min)
check(fig, ax, "S2")
fig.savefig(OUT / "figS2_vit_architecture.pdf")
fig.savefig(OUT / "figS2_vit_architecture.png", dpi=600)
plt.close(fig)
