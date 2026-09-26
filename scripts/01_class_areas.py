from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

IN = Path(os.environ.get("JAXA_DIR", ".")).expanduser()
BOUNDARY = Path(os.environ.get("VN_BOUNDARY", "")).expanduser()
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


BOUNDARY = BOUNDARY if BOUNDARY.is_file() else find_file("vnm_admin0.shp", Path(os.environ.get("HDX_DIR", ".")).expanduser())


L1 = {1: "Residence", 3: "Rice paddies", 4: "Cropland", 7: "Grassland", 8: "Barren land",
      9: "Scrubland", 10: "Forest", 15: "Wetland", 18: "Open water", 19: "Aquaculture"}
L2 = {1: "Residence 1", 2: "Residence 2", 3: "Rice paddies", 4: "Woody crops", 5: "Other crops",
      6: "In-house crops", 7: "Grassland", 8: "Barren land", 9: "Scrubland",
      10: "Deciduous broadleaf forest", 11: "Evergreen broadleaf forest",
      12: "Evergreen needleleaf forest", 14: "Plantation", 15: "Mangrove forest",
      16: "Inland wetland", 18: "Open water", 19: "Aquaculture", 20: "Bamboo"}


def pixel_area_km2(transform: rasterio.Affine, nrows: int) -> np.ndarray:
    r = 6371.0072
    lat_top = transform.f + np.arange(nrows) * transform.e
    lat_bot = lat_top + transform.e
    dlon = np.radians(abs(transform.a))
    return (r**2) * dlon * np.abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot)))


rows = []
_mask: np.ndarray | None = None


def vn_mask(src: rasterio.io.DatasetReader) -> np.ndarray:
    global _mask
    if _mask is None:
        adm = gpd.read_file(BOUNDARY).to_crs(src.crs)
        _mask = rasterize(((g, 1) for g in adm.geometry), out_shape=src.shape, transform=src.transform, dtype="uint8") == 1
    return _mask


for level, names in (("L1", L1), ("L2", L2)):
    for f in sorted(str(p) for p in IN.rglob(f"VLUCD_{level}_250m_*.tif")):
        year = int(f[-8:-4])
        with rasterio.open(f) as src:
            a = src.read(1)
            a = np.where(vn_mask(src), a, 0)
            area_row = pixel_area_km2(src.transform, src.height)
        area = np.broadcast_to(area_row[:, None], a.shape)
        s = np.bincount(a.ravel(), weights=area.ravel(), minlength=256)
        n = np.bincount(a.ravel(), minlength=256)
        for code, name in names.items():
            rows.append({"level": level, "year": year, "code": code, "class": name,
                         "pixels": int(n[code]), "area_km2": float(s[code])})
        extra = set(np.flatnonzero(n)) - set(names) - {0}
        assert not extra, (f, extra)
df = pd.DataFrame(rows)
df.to_csv(OUT / "class_areas_km2.csv", index=False)
print(df[df.level == "L1"].pivot(index="year", columns="class", values="area_km2").round(0).loc[[1990, 2000, 2010, 2020]].T)
print(df[df.level == "L2"].pivot(index="year", columns="class", values="area_km2").round(0).loc[[1990, 2000, 2010, 2020]].T)
