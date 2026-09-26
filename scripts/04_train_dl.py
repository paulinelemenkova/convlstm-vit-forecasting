from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from rasterio.features import rasterize
from torch import nn

IN = Path(os.environ.get("JAXA_DIR", ".")).expanduser()
ADM0 = Path(os.environ.get("VN_ADMIN0", "")).expanduser()
PRED = Path(os.environ.get("PRED_DIR", "./hindcast"))


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


GROUPS = {
    1: ("Natural forest", [10, 11, 12, 20]),
    2: ("Plantation", [14]),
    3: ("Mangrove", [15]),
    4: ("Woody crops", [4]),
    5: ("Other agriculture", [3, 5, 6]),
    6: ("Built-up", [1, 2]),
    7: ("Other land", [7, 8, 9, 16, 18, 19]),
}
K, T_IN, LEAD, TILE, BLOCK = 7, 5, 10, 64, 4
LUT = np.zeros(256, dtype=np.uint8)
for gid, (_gname, codes) in GROUPS.items():
    LUT[codes] = gid

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["convlstm", "vit"], required=True)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                help="auto = CUDA, else Apple GPU (mps), else CPU")
args = ap.parse_args()
if args.smoke:
    args.seeds, args.epochs = 2, 1
    PRED = PRED / "smoke"
PRED.mkdir(parents=True, exist_ok=True)
if args.device == "auto":
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
else:
    dev = torch.device(args.device)
tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if tv < (2, 3) and int(np.__version__.split(".")[0]) >= 2:
    raise SystemExit(f"PyTorch {torch.__version__} does not work with NumPy {np.__version__}: "
                     "run  pip install -U torch  (or  pip install 'numpy<2')")
print(f"device: {dev}; torch {torch.__version__}; {platform.processor() or platform.machine()}")

with rasterio.open(find_file("VLUCD_L2_250m_2010.tif", IN)) as src:
    profile, shape, transform, crs = src.profile.copy(), src.shape, src.transform, src.crs
inside = rasterize(((g, 1) for g in gpd.read_file(ADM0).to_crs(crs).geometry), out_shape=shape,
                   transform=transform, dtype="uint8") == 1
YEARS = list(range(1990, 2021))
stack = np.zeros((len(YEARS), *shape), dtype=np.uint8)
for i, y in enumerate(YEARS):
    with rasterio.open(find_file(f"VLUCD_L2_250m_{y}.tif", IN)) as src:
        stack[i] = LUT[src.read(1)]
valid = inside & (stack > 0).all(axis=0)
stack[:, ~valid] = 0
if args.smoke:
    r0, c0 = int((transform.f - 14.5) / -transform.e), int((107.0 - transform.c) / transform.a)
    win = (slice(r0, r0 + 6 * TILE), slice(c0, c0 + 6 * TILE))
else:
    rows_, cols_ = np.flatnonzero(valid.any(1)), np.flatnonzero(valid.any(0))
    win = (slice(rows_[0], rows_[-1] + 1), slice(cols_[0], cols_[-1] + 1))
stack, valid = stack[:, win[0], win[1]], valid[win]
H, W = valid.shape
Hp, Wp = -(-H // TILE) * TILE, -(-W // TILE) * TILE
stack = np.pad(stack, ((0, 0), (0, Hp - H), (0, Wp - W)))
valid = np.pad(valid, ((0, Hp - H), (0, Wp - W)))
yi = {y: i for i, y in enumerate(YEARS)}

tiles = [(r, c) for r in range(0, Hp, TILE) for c in range(0, Wp, TILE)
         if valid[r:r + TILE, c:c + TILE].mean() >= 0.1]
rng = np.random.default_rng(42)
blocks = sorted({(r // (TILE * BLOCK), c // (TILE * BLOCK)) for r, c in tiles})
val_blocks = {blocks[i] for i in rng.choice(len(blocks), max(1, round(0.2 * len(blocks))), replace=False)}
tr_tiles = [t for t in tiles if (t[0] // (TILE * BLOCK), t[1] // (TILE * BLOCK)) not in val_blocks]
va_tiles = [t for t in tiles if (t[0] // (TILE * BLOCK), t[1] // (TILE * BLOCK)) in val_blocks]
T_TRAIN = list(range(1994, 2001))
print(f"grid {H} x {W}; tiles {len(tiles)} (training {len(tr_tiles)}, validation {len(va_tiles)}); "
      f"samples per epoch {len(tr_tiles) * len(T_TRAIN)}")

freq = np.bincount(stack[[yi[t + LEAD] for t in T_TRAIN]].ravel(), minlength=K + 1)[1:].astype(float)
cls_w = np.clip(np.sqrt(np.median(freq) / np.maximum(freq, 1)), 0.2, 10.0)
print("class weights:", dict(zip([v[0] for v in GROUPS.values()], cls_w.round(2), strict=True)))


def sample(r: int, c: int, t: int, with_target: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = stack[yi[t] - T_IN + 1:yi[t] + 1, r:r + TILE, c:c + TILE].astype(np.int64)
    xo = F.one_hot(torch.from_numpy(x), K + 1)[..., 1:].permute(0, 3, 1, 2).float()
    if not with_target:
        return xo, None
    yv = stack[yi[t + LEAD], r:r + TILE, c:c + TILE].astype(np.int64) - 1
    return xo, torch.from_numpy(yv)


def batches(tl: list[tuple[int, int]], shuffle: bool, gen: np.random.Generator):
    items = [(r, c, t) for r, c in tl for t in T_TRAIN]
    if shuffle:
        gen.shuffle(items)
    for s in range(0, len(items), args.batch):
        xs, ys = zip(*(sample(*it) for it in items[s:s + args.batch]), strict=True)
        yield torch.stack(xs), torch.stack(ys)


class ConvLSTMCell(nn.Module):
    def __init__(self, cin: int, ch: int) -> None:
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(cin + ch, 4 * ch, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTM(nn.Module):
    def __init__(self, ch: int = 64) -> None:
        super().__init__()
        self.c1, self.c2 = ConvLSTMCell(K, ch), ConvLSTMCell(ch, ch)
        self.head = nn.Conv2d(ch, K, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, h, w = x.shape
        z = x.new_zeros(b, self.c1.ch, h, w)
        h1, s1, h2, s2 = z, z, z, z
        for t in range(x.shape[1]):
            h1, s1 = self.c1(x[:, t], h1, s1)
            h2, s2 = self.c2(h1, h2, s2)
        return self.head(h2)


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
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.tpos, std=0.02)
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

    def rollout(self) -> torch.Tensor:
        n = self.blocks[0].last_att.shape[-1]
        eye = torch.eye(n, device=self.pos.device)
        r = eye.expand_as(self.blocks[0].last_att)
        for blk in self.blocks:
            a = 0.5 * blk.last_att + 0.5 * eye
            r = (a / a.sum(-1, keepdim=True)) @ r
        rec = r.mean(1)
        return rec.reshape(rec.shape[0], T_IN, -1).sum(-1)


def make() -> nn.Module:
    return ConvLSTM() if args.model == "convlstm" else TemporalViT()


weights = torch.tensor(cls_w, dtype=torch.float32, device=dev)
log, states = [], []
t_start = time.perf_counter()
for seed in range(args.seeds):
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    net = make().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr or (1e-3 if args.model == "convlstm" else 3e-4),
                            weight_decay=1e-4)
    best, best_state, wait = np.inf, None, 0
    for ep in range(args.epochs):
        net.train()
        tl, nb = 0.0, 0
        for xb, yb in batches(tr_tiles, True, gen):
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(net(xb), yb, weight=weights, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl, nb = tl + loss.item(), nb + 1
        net.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in batches(va_tiles, False, gen):
                xb, yb = xb.to(dev), yb.to(dev)
                vl, vn = vl + F.cross_entropy(net(xb), yb, weight=weights, ignore_index=-1).item(), vn + 1
        vl /= max(vn, 1)
        log.append({"seed": seed, "epoch": ep + 1, "train_loss": tl / max(nb, 1), "val_loss": vl})
        print(f"seed {seed} epoch {ep + 1}: train {tl / max(nb, 1):.4f}  val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    assert best_state is not None
    states.append(best_state)
    torch.save(best_state, PRED / f"{args.model}_seed{seed}.pt")
t_train = time.perf_counter() - t_start
pd.DataFrame(log).to_csv(PRED / f"{args.model}_train_log.csv", index=False)


def predict(t_last: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    stride = TILE // 2
    ramp = np.minimum(np.arange(TILE) + 1, np.arange(TILE)[::-1] + 1).astype(np.float32)
    wtile = torch.from_numpy(np.outer(ramp, ramp))
    votes = np.zeros((args.seeds, Hp, Wp), dtype=np.uint8)
    prob_sum = torch.zeros(K, Hp, Wp)
    att: list[np.ndarray] = []
    pos = [(r, c) for r in range(0, Hp - TILE + 1, stride) for c in range(0, Wp - TILE + 1, stride)
           if valid[r:r + TILE, c:c + TILE].any()]
    for s, st in enumerate(states):
        net = make().to(dev)
        net.load_state_dict(st)
        net.eval()
        acc = torch.zeros(K, Hp, Wp)
        with torch.no_grad():
            for b0 in range(0, len(pos), args.batch):
                chunk = pos[b0:b0 + args.batch]
                xb = torch.stack([sample(r, c, t_last, with_target=False)[0] for r, c in chunk]).to(dev)
                keep = isinstance(net, TemporalViT)
                pr = torch.softmax(net(xb, keep) if keep else net(xb), 1).cpu()
                if keep:
                    att.append(net.rollout().cpu().numpy())
                for (r, c), p_ in zip(chunk, pr, strict=True):
                    acc[:, r:r + TILE, c:c + TILE] += p_ * wtile
        acc /= acc.sum(0, keepdim=True).clamp_min(1e-9)
        votes[s] = acc.argmax(0).numpy() + 1
        prob_sum += acc
    ens = (prob_sum.argmax(0).numpy() + 1).astype(np.uint8)
    spread = (100 * (votes != ens[None]).mean(0)).astype(np.uint8)
    ens[~valid], spread[~valid] = 0, 0
    return ens[:H, :W], spread[:H, :W], (np.concatenate(att).mean(0) if att else None)


def save(name: str, a: np.ndarray) -> None:
    full = np.zeros(shape, dtype=np.uint8)
    full[win] = a
    prof = profile | {"dtype": "uint8", "nodata": 0, "count": 1, "compress": "deflate"}
    with rasterio.open(PRED / f"{name}.tif", "w", **prof) as dst:
        dst.write(full, 1)


t0 = time.perf_counter()
p20, s20, att20 = predict(2010)
t_pred = time.perf_counter() - t0
p30, s30, _att30 = predict(2020)
for n, a in ((f"{args.model}_2020", p20), (f"{args.model}_2020_spread", s20), (f"{args.model}_2030", p30),
             (f"{args.model}_2030_spread", s30)):
    save(n, a)
if att20 is not None:
    pd.DataFrame({"input_year": list(range(2006, 2011)), "attention_share": att20 / att20.sum()}).to_csv(
        PRED / f"{args.model}_attention.csv", index=False, float_format="%.4f")

r = 6371.0072
lat_top = transform.f + (np.arange(shape[0])[win[0]]) * transform.e
area_row = r**2 * np.radians(abs(transform.a)) * np.abs(np.sin(np.radians(lat_top)) -
                                                        np.sin(np.radians(lat_top + transform.e)))
v = valid[:H, :W]
w = np.broadcast_to(area_row[:, None], (H, W))[v]
o10, o20, p = stack[yi[2010], :H, :W][v], stack[yi[2020], :H, :W][v], p20[v]
cm = np.bincount((o20.astype(int) - 1) * K + (p - 1), weights=w, minlength=K * K).reshape(K, K)
pm = cm / cm.sum()
oa = np.trace(pm)
pe = (pm.sum(0) * pm.sum(1)).sum()
q = 0.5 * np.abs(pm.sum(0) - pm.sum(1)).sum()
fo, fp = np.isin(o20, (1, 2, 3)), np.isin(p, (1, 2, 3))
ch_o, ch_p = o20 != o10, p != o10
hits, wrong = w[ch_o & ch_p & (p == o20)].sum(), w[ch_o & ch_p & (p != o20)].sum()
miss, fa = w[ch_o & ~ch_p].sum(), w[~ch_o & ch_p].sum()
label = {"convlstm": "ConvLSTM", "vit": "Temporal ViT"}[args.model]
row = {"Model": label, "OA": oa, "Kappa": (oa - pe) / (1 - pe), "Quantity": q, "Allocation": 1 - oa - q,
       "F1 forest": 2 * w[fo & fp].sum() / (w[fo].sum() + w[fp].sum()),
       "F1 non-forest": 2 * w[~fo & ~fp].sum() / (w[~fo].sum() + w[~fp].sum()),
       "FoM": hits / (hits + wrong + miss + fa), "hits_km2": hits, "wrong_km2": wrong, "miss_km2": miss,
       "false_alarm_km2": fa, "Time_s": t_train + t_pred}
mf = PRED / "hindcast_metrics.csv"
met = pd.read_csv(mf) if mf.exists() else pd.DataFrame()
met = pd.concat([met[met.get("Model", pd.Series(dtype=str)) != label] if len(met) else met, pd.DataFrame([row])])
met.to_csv(mf, index=False, float_format="%.4f")
f1 = PRED / "hindcast_f1_groups.csv"
fr = pd.read_csv(f1) if f1.exists() else pd.DataFrame(columns=["Model", "Group", "F1"])
fr = fr[fr.Model != label]
new = [{"Model": label, "Group": GROUPS[k][0],
        "F1": 2 * cm[k - 1, k - 1] / max(cm[k - 1].sum() + cm[:, k - 1].sum(), 1e-9)} for k in range(1, K + 1)]
pd.concat([fr, pd.DataFrame(new)]).to_csv(f1, index=False, float_format="%.4f")
cmf = pd.DataFrame(cm / 1000, index=[v_[0] for v_ in GROUPS.values()], columns=[v_[0] for v_ in GROUPS.values()])
cmf.to_csv(PRED / f"{args.model}_confusion_2020_1000km2.csv", float_format="%.3f")
print(pd.DataFrame([row]).round(4).to_string(index=False))
print(f"training {t_train / 60:.1f} min, 2020 prediction {t_pred / 60:.1f} min, device {dev}")
