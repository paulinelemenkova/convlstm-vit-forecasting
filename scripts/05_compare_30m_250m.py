from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

IN = Path(os.environ.get("JAXA_DIR", ".")).expanduser()
ADM0 = Path(os.environ.get("VN_ADMIN0", "")).expanduser()
OUT = Path(os.environ.get("OUT_DIR", ".")).expanduser()
PAT = {"30m": os.environ.get("PATTERN_30M", "*L1*30m*{year}*.tif"), "250m": "VLUCD_L1_250m_{year}.tif"}
L1 = {1: "Residence", 3: "Rice paddies", 4: "Cropland", 7: "Grassland", 8: "Barren land", 9: "Scrubland",
      10: "Forest", 15: "Wetland", 18: "Open water", 19: "Aquaculture"}
STRIP = 1024


def find_file(pattern: str, root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"folder does not exist: {root} (check the spelling; Finder may hide an extension)")
    hits = sorted(p for p in root.rglob(pattern) if not p.name.startswith("._"))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern} below {root.resolve()}; set JAXA_DIR (and HDX_DIR)")
    return hits[0]


ADM0 = ADM0 if ADM0.is_file() else find_file("vnm_admin0.shp", Path(os.environ.get("HDX_DIR", ".")).expanduser())
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--years", type=int, nargs="*", default=list(range(1990, 2021)))
args = ap.parse_args()
years_req = sorted(set(args.years) | {1990, 2020})
years, missing = [], []
for y_ in years_req:
    try:
        find_file(PAT["30m"].format(year=y_), IN)
        years.append(y_)
    except FileNotFoundError:
        missing.append(y_)
if missing:
    print(f"30 m map missing for {missing}: these years are skipped")
if 1990 not in years or 2020 not in years:
    raise SystemExit("the 30 m maps of 1990 and 2020 are required")


def row_area_km2(transform: rasterio.Affine, row0: int, nrows: int) -> np.ndarray:
    r = 6371.0072
    lat_top = transform.f + (row0 + np.arange(nrows)) * transform.e
    lat_bot = lat_top + transform.e
    return r**2 * np.radians(abs(transform.a)) * np.abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot)))


_masks: dict[tuple, list[np.ndarray]] = {}
geom_cache: dict[str, list] = {}


def class_areas(path: Path) -> np.ndarray:
    out = np.zeros(256)
    with rasterio.open(path) as src:
        key = (src.transform, src.width, src.height, str(src.crs))
        if str(src.crs) not in geom_cache:
            geom_cache[str(src.crs)] = list(gpd.read_file(ADM0).to_crs(src.crs).geometry)
        geoms = geom_cache[str(src.crs)]
        build = key not in _masks
        if build:
            _masks[key] = []
        for i, row0 in enumerate(range(0, src.height, STRIP)):
            n = min(STRIP, src.height - row0)
            win = Window(0, row0, src.width, n)
            if build:
                m = rasterize(((g, 1) for g in geoms), out_shape=(n, src.width),
                              transform=src.window_transform(win), dtype="uint8")
                _masks[key].append(np.packbits(m.astype(bool), axis=None))
            inside = np.unpackbits(_masks[key][i], count=n * src.width).astype(bool).reshape(n, src.width)
            if not inside.any():
                continue
            a = src.read(1, window=win)
            w = np.broadcast_to(row_area_km2(src.transform, row0, n)[:, None], a.shape)
            out += np.bincount(a[inside], weights=w[inside], minlength=256)
    return out


rows = []
for y in years:
    for ver in ("30m", "250m"):
        t0 = time.perf_counter()
        f = find_file(PAT[ver].format(year=y), IN)
        s = class_areas(f)
        extra = sorted(set(np.flatnonzero(s)) - set(L1) - {0})
        if extra:
            print(f"  warning: unexpected codes {extra} in {f.name}")
        for code, name in L1.items():
            rows.append({"year": y, "version": ver, "code": code, "class": name, "area_km2": s[code]})
        print(f"{y} {ver:>4}: {f.name}  forest {s[10]:9.0f} km2  total {s[list(L1)].sum():9.0f} km2  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "l1_areas_30m_250m.csv", index=False, float_format="%.2f")

d = pd.DataFrame(rows)
w = d.pivot_table(index=["class", "code"], columns=["version", "year"], values="area_km2")
res = []
for (name, code), r in w.iterrows():
    a30_90, a30_20, a250_90, a250_20 = (r[("30m", 1990)], r[("30m", 2020)], r[("250m", 1990)], r[("250m", 2020)])
    s30 = np.array([r[("30m", y)] for y in years])
    s250 = np.array([r[("250m", y)] for y in years])
    res.append({"code": code, "class": name, "30m_1990": a30_90, "250m_1990": a250_90,
                "diff_1990_pct": 100 * (a250_90 - a30_90) / a30_90, "30m_2020": a30_20, "250m_2020": a250_20,
                "diff_2020_pct": 100 * (a250_20 - a30_20) / a30_20,
                "net_30m": a30_20 - a30_90, "net_250m": a250_20 - a250_90,
                "r_annual": np.corrcoef(s30, s250)[0, 1] if len(years) > 2 else np.nan})
cmp_ = pd.DataFrame(res).sort_values("code")
sums = {c: float(cmp_[c].sum()) for c in ("30m_1990", "250m_1990", "30m_2020", "250m_2020", "net_30m", "net_250m")}
sums["diff_1990_pct"] = 100 * (sums["250m_1990"] - sums["30m_1990"]) / sums["30m_1990"]
sums["diff_2020_pct"] = 100 * (sums["250m_2020"] - sums["30m_2020"]) / sums["30m_2020"]
tot: dict[str, object] = {"code": 0, "class": "Total", **sums}
cmp_ = pd.concat([cmp_, pd.DataFrame([tot])], ignore_index=True)
cmp_.to_csv(OUT / "l1_comparison_30m_250m.csv", index=False, float_format="%.3f")


def k(v: float) -> str:
    return f"{v / 1000:.1f}".replace("-", "$-$")


def pct(v: float) -> str:
    return (f"{v:+.0f}".replace("-", "$-$").replace("+", "$+$")) + "\\%"


lines = [f"{r['class']} & {k(r['30m_1990'])} & {k(r['250m_1990'])} & {pct(r['diff_1990_pct'])} & "
         f"{k(r['30m_2020'])} & {k(r['250m_2020'])} & {pct(r['diff_2020_pct'])} & {k(r['net_30m'])} & "
         f"{k(r['net_250m'])} & " + ("--" if np.isnan(r["r_annual"]) else f"{r['r_annual']:.2f}") + " \\\\"
         for _, r in cmp_.iterrows()]
(OUT / "l1_comparison_table.tex").write_text("\n".join(lines) + "\n")
pd.set_option("display.width", 200)
print(cmp_.round(2).to_string(index=False))
print(f"years processed: {years}; skipped (no 30 m map): {missing}")
