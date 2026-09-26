from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

IN = Path(os.environ.get("JAXA_DIR", ".")).expanduser()
ADM0 = Path(os.environ.get("VN_ADMIN0", "")).expanduser()
ADM1 = Path(os.environ.get("VN_ADMIN1", "")).expanduser()
OUT = Path(__file__).parent


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


GROUPS = {
    1: ("Natural forest", [10, 11, 12, 20]),
    2: ("Plantation", [14]),
    3: ("Mangrove", [15]),
    4: ("Woody crops", [4]),
    5: ("Other agriculture", [3, 5, 6]),
    6: ("Built-up", [1, 2]),
    7: ("Other land", [7, 8, 9, 16, 18, 19]),
}
NAMES = {k: v[0] for k, v in GROUPS.items()}
LUT = np.zeros(256, dtype=np.uint8)
for g, (_name, codes) in GROUPS.items():
    LUT[codes] = g
YEARS = (1990, 2000, 2010, 2020)
PERIODS = ((1990, 2000), (2000, 2010), (2010, 2020), (1990, 2020))


def path(year: int) -> Path:
    return find_file(f"VLUCD_L2_250m_{year}.tif", IN)


def pixel_area_km2(transform: rasterio.Affine, nrows: int) -> np.ndarray:
    r = 6371.0072
    lat_top = transform.f + np.arange(nrows) * transform.e
    lat_bot = lat_top + transform.e
    return r**2 * np.radians(abs(transform.a)) * np.abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot)))


with rasterio.open(path(1990)) as src:
    shape, transform, crs = src.shape, src.transform, src.crs
area = np.broadcast_to(pixel_area_km2(transform, shape[0])[:, None], shape)
adm0 = gpd.read_file(ADM0).to_crs(crs)
inside = rasterize(((g, 1) for g in adm0.geometry), out_shape=shape, transform=transform, dtype="uint8") == 1
adm1 = gpd.read_file(ADM1).to_crs(crs).reset_index(drop=True)
prov = rasterize(((g, i + 1) for i, g in enumerate(adm1.geometry)), out_shape=shape, transform=transform,
                 dtype="uint8")

maps: dict[int, np.ndarray] = {}
for y in YEARS:
    with rasterio.open(path(y)) as src:
        maps[y] = np.where(inside, LUT[src.read(1)], 0).astype(np.uint8)

rows = []
for y0, y1 in PERIODS:
    code = maps[y0].astype(np.int32) * 8 + maps[y1]
    ok = (maps[y0] > 0) & (maps[y1] > 0)
    s = np.bincount(code[ok], weights=area[ok], minlength=64).reshape(8, 8)
    for a in NAMES:
        for b in NAMES:
            rows.append({"period": f"{y0}-{y1}", "from": NAMES[a], "to": NAMES[b], "area_km2": s[a, b]})
tr = pd.DataFrame(rows)
tr.to_csv(OUT / "transitions_km2.csv", index=False, float_format="%.1f")

gn = []
for per_name, t in tr.groupby("period", sort=False):
    m = t.pivot(index="from", columns="to", values="area_km2").loc[list(NAMES.values()), list(NAMES.values())]
    for grp in NAMES.values():
        loss = m.loc[grp].sum() - m.loc[grp, grp]
        gain = m[grp].sum() - m.loc[grp, grp]
        gn.append({"period": per_name, "group": grp, "start": m.loc[grp].sum(), "end": m[grp].sum(), "persist": m.loc[grp, grp],
                   "gross_loss": loss, "gross_gain": gain, "net": gain - loss, "swap": 2 * min(loss, gain)})
gn_df = pd.DataFrame(gn)
gn_df.to_csv(OUT / "gross_net_km2.csv", index=False, float_format="%.1f")

pr = []
for i, r in adm1.iterrows():
    sel = prov == i + 1
    rec: dict[str, object] = {"province": r["adm1_name"], "pcode": r["adm1_pcode"]}
    for y in (1990, 2020):
        m = maps[y][sel]
        a = area[sel]
        rec[f"natural_{y}"] = a[m == 1].sum()
        rec[f"plantation_{y}"] = a[m == 2].sum()
        rec[f"forest_{y}"] = a[(m >= 1) & (m <= 3)].sum()
    m0, m1, a = maps[1990][sel], maps[2020][sel], area[sel]
    rec["natural_loss"] = a[(m0 == 1) & (m1 != 1)].sum()
    rec["natural_gain"] = a[(m0 != 1) & (m1 == 1)].sum()
    pr.append(rec)
pr_df = pd.DataFrame(pr)
for k in ("natural", "plantation", "forest"):
    pr_df[f"{k}_net"] = pr_df[f"{k}_2020"] - pr_df[f"{k}_1990"]
pr_df.to_csv(OUT / "province_change_km2.csv", index=False, float_format="%.1f")

pd.set_option("display.width", 200)
print(gn_df.round(0).to_string(index=False))
m = tr[tr.period == "1990-2020"].pivot(index="from", columns="to", values="area_km2")
print((m.loc[list(NAMES.values()), list(NAMES.values())] / 1000).round(1).to_string())
cols = ["province", "natural_net", "plantation_net", "forest_net", "natural_loss", "natural_gain"]
print(pr_df.sort_values("forest_net")[cols].round(0).to_string(index=False))
