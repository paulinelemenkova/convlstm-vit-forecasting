from __future__ import annotations

import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier

IN = Path(os.environ.get("JAXA_DIR", ".")).expanduser()
ADM0 = Path(os.environ.get("VN_ADMIN0", "")).expanduser()
PRED = Path(os.environ.get("PRED_DIR", "./hindcast"))
PRED.mkdir(parents=True, exist_ok=True)


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
K = 7
LUT = np.zeros(256, dtype=np.uint8)
for gid, (_gname, codes) in GROUPS.items():
    LUT[codes] = gid
rng = np.random.default_rng(0)


def src_path(year: int) -> Path:
    return find_file(f"VLUCD_L2_250m_{year}.tif", IN)


with rasterio.open(src_path(2010)) as src:
    profile = src.profile.copy()
    shape, transform, crs = src.shape, src.transform, src.crs
inside = rasterize(((g, 1) for g in gpd.read_file(ADM0).to_crs(crs).geometry), out_shape=shape,
                   transform=transform, dtype="uint8") == 1


def groups(year: int) -> np.ndarray:
    with rasterio.open(src_path(year)) as s:
        return np.where(inside, LUT[s.read(1)], 0).astype(np.uint8)


def pixel_area_km2() -> np.ndarray:
    r = 6371.0072
    lat_top = transform.f + np.arange(shape[0]) * transform.e
    lat_bot = lat_top + transform.e
    a = r**2 * np.radians(abs(transform.a)) * np.abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot)))
    return np.broadcast_to(a[:, None], shape)


def shares(g: np.ndarray, size: int) -> np.ndarray:
    valid = uniform_filter((g > 0).astype(np.float32), size, mode="constant")
    out = np.empty((K, *g.shape), dtype=np.float32)
    for k in range(1, K + 1):
        out[k - 1] = uniform_filter((g == k).astype(np.float32), size, mode="constant") / np.maximum(valid, 1e-6)
    return out


years = list(range(1990, 2011)) + [2020]
maps = {y: groups(y) for y in years}
valid = np.logical_and.reduce([maps[y] > 0 for y in years])
print(f"cells with data in some but not all years: {int((np.logical_or.reduce([maps[y] > 0 for y in years]) & ~valid).sum())}")
for y in years:
    maps[y][~valid] = 0
idx = np.flatnonzero(valid)
area = pixel_area_km2()
w = area.ravel()[idx]
obs10 = maps[2010].ravel()[idx]
obs20 = maps[2020].ravel()[idx]
print(f"valid cells: {idx.size:,}")


def save(name: str, pred: np.ndarray) -> None:
    full = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    full[idx] = pred
    prof = profile | {"dtype": "uint8", "nodata": 0, "count": 1, "compress": "deflate"}
    with rasterio.open(PRED / f"{name}_2020.tif", "w", **prof) as dst:
        dst.write(full.reshape(shape), 1)


preds: dict[str, np.ndarray] = {}
times: dict[str, float] = {}

t0 = time.perf_counter()
preds["Persistence"] = obs10.copy()
times["Persistence"] = time.perf_counter() - t0

t0 = time.perf_counter()
g00 = maps[2000].ravel()[idx]
T = np.bincount((g00.astype(int) - 1) * K + (obs10 - 1), weights=w, minlength=K * K).reshape(K, K)
P = T / T.sum(axis=1, keepdims=True)
a10 = np.bincount(obs10 - 1, weights=w, minlength=K)
demand = a10 @ P
nb = shares(maps[2010], 5).reshape(K, -1)[:, idx]
suit = P[obs10 - 1].T * (0.01 + nb)
del nb
cw = np.ones(K)
for _ in range(60):
    lab = np.argmax(suit * cw[:, None], axis=0)
    got = np.bincount(lab, weights=w, minlength=K)
    cw *= np.sqrt(demand / np.maximum(got, 1e-3))
    if np.max(np.abs(got - demand) / demand) < 0.005:
        break
preds["CA–Markov"] = (lab + 1).astype(np.uint8)
del suit
times["CA–Markov"] = time.perf_counter() - t0
print("CA-Markov demand vs allocated (10^3 km2):")
print(pd.DataFrame({"demand": demand / 1e3, "allocated": got / 1e3}, index=[v[0] for v in GROUPS.values()]).round(2))


def features(t: int, cells: np.ndarray) -> np.ndarray:
    onehot = [np.eye(K, dtype=np.float32)[maps[y].ravel()[cells] - 1] for y in range(t - 4, t + 1)]
    sh = shares(maps[t], 3).reshape(K, -1)[:, cells].T
    return np.hstack([*onehot, sh])


t0 = time.perf_counter()
Xs, ys = [], []
for t in range(1994, 2001):
    cells = rng.choice(idx, 50_000, replace=False)
    Xs.append(features(t, cells))
    ys.append(maps[t + 10].ravel()[cells])
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_features="sqrt", n_jobs=2, random_state=0)
rf.fit(np.vstack(Xs), np.concatenate(ys))
pred_rf = np.empty(idx.size, dtype=np.uint8)
for s in range(0, idx.size, 500_000):
    pred_rf[s:s + 500_000] = rf.predict(features(2010, idx[s:s + 500_000]))
preds["Random Forest"] = pred_rf
times["Random Forest"] = time.perf_counter() - t0


def metrics(p: np.ndarray) -> dict[str, float]:
    cm = np.bincount((obs20.astype(int) - 1) * K + (p - 1), weights=w, minlength=K * K).reshape(K, K)
    n = cm.sum()
    pm = cm / n
    oa = np.trace(pm)
    pe = (pm.sum(0) * pm.sum(1)).sum()
    q = 0.5 * np.abs(pm.sum(0) - pm.sum(1)).sum()
    fo, fp = np.isin(obs20, (1, 2, 3)), np.isin(p, (1, 2, 3))

    def f1(o: np.ndarray, q_: np.ndarray) -> float:
        tp = w[o & q_].sum()
        return 2 * tp / (w[o].sum() + w[q_].sum())

    ch_o, ch_p = obs20 != obs10, p != obs10
    hits = w[ch_o & ch_p & (p == obs20)].sum()
    wrong = w[ch_o & ch_p & (p != obs20)].sum()
    miss = w[ch_o & ~ch_p].sum()
    fa = w[~ch_o & ch_p].sum()
    return {"OA": oa, "Kappa": (oa - pe) / (1 - pe), "Quantity": q, "Allocation": 1 - oa - q,
            "F1 forest": f1(fo, fp), "F1 non-forest": f1(~fo, ~fp),
            "FoM": hits / (hits + wrong + miss + fa), "hits_km2": hits, "wrong_km2": wrong, "miss_km2": miss,
            "false_alarm_km2": fa}


rows, f1rows = [], []
for name, p in preds.items():
    m = metrics(p)
    m["Time_s"] = times[name]
    rows.append({"Model": name, **m})
    for k in range(1, K + 1):
        o, q_ = obs20 == k, p == k
        f1rows.append({"Model": name, "Group": GROUPS[k][0],
                       "F1": 2 * w[o & q_].sum() / (w[o].sum() + w[q_].sum())})
    save({"Persistence": "persistence", "CA–Markov": "camarkov", "Random Forest": "rf"}[name], p)
met = pd.DataFrame(rows)
met.to_csv(PRED / "hindcast_metrics.csv", index=False, float_format="%.4f")
pd.DataFrame(f1rows).to_csv(PRED / "hindcast_f1_groups.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 200)
print(met.round(4).to_string(index=False))
print(pd.DataFrame(f1rows).pivot(index="Group", columns="Model", values="F1").round(3))
print(f"observed change 2010-2020: {w[obs20 != obs10].sum():.0f} km2 of {w.sum():.0f} km2")
