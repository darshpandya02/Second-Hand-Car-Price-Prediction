"""Loading and cleaning of the CarDekho vehicle dataset (Kaggle: nehalbirla/vehicle-dataset-from-cardekho)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
V3_PATH = RAW_DIR / "car_details_v3.csv"
LEGACY_PATH = RAW_DIR / "car_data.csv"

# SHA-256 of the files as downloaded from Kaggle (dataset version 4, updated 2023-01-14).
CHECKSUMS = {
    "car_details_v3.csv": "1b2a30b2f36fd246366773fe5375399692f1a113890ad2a6d464b8c8deaadadb",
    "car_data.csv": "00b477def120548e583c638b00146982956b80646817901245ef78da9c4b3063",
}

TARGET = "selling_price"
CATEGORICAL = ["brand", "model", "fuel", "seller_type", "transmission"]
NUMERIC = ["year", "km_driven", "owner", "mileage", "engine", "max_power", "seats"]
FEATURES = CATEGORICAL + NUMERIC

# Brands whose name spans two tokens in the "name" column.
TWO_WORD_BRANDS = {"Land": "Land Rover", "Ashok": "Ashok Leyland"}

OWNER_MAP = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path) -> None:
    expected = CHECKSUMS.get(path.name)
    if expected is not None and sha256(path) != expected:
        raise ValueError(f"checksum mismatch for {path}")


def split_name(name: str) -> tuple[str, str]:
    """Split a listing title such as 'Maruti Swift Dzire VDI' into (brand, model)."""
    tokens = name.split()
    first = tokens[0]
    if first in TWO_WORD_BRANDS and len(tokens) > 1:
        brand, rest = TWO_WORD_BRANDS[first], tokens[2:]
    else:
        brand, rest = first, tokens[1:]
    # "Mercedes-Benz New C-Class": skip the marketing prefix.
    if rest and rest[0] == "New" and len(rest) > 1:
        rest = rest[1:]
    model = rest[0] if rest else "Unknown"
    return brand, model


def _leading_number(s: pd.Series) -> pd.Series:
    extracted = s.astype(str).str.extract(r"([\d.]+)")[0]
    return pd.to_numeric(extracted, errors="coerce").astype(float)


def clean_v3(raw: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Turn the raw 'Car details v3.csv' frame into model-ready columns.

    - brand/model parsed from the listing title
    - owner mapped to an ordinal count (Test Drive Car = 0)
    - mileage (kmpl or km/kg), engine (CC), max_power (bhp) parsed to floats
    - torque is dropped: it mixes Nm/kgm units and rpm ranges in free text
    - exact duplicate rows are removed so that copies cannot land in both train and test
    """
    df = raw.copy()
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    parts = df["name"].map(split_name)
    df["brand"] = parts.map(lambda p: p[0])
    df["model"] = parts.map(lambda p: p[1])
    df["owner"] = df["owner"].map(OWNER_MAP).astype(float)
    for col in ("mileage", "engine", "max_power"):
        df[col] = _leading_number(df[col])
    df["seats"] = pd.to_numeric(df["seats"], errors="coerce")
    df["year"] = df["year"].astype(float)
    df["km_driven"] = df["km_driven"].astype(float)
    df[TARGET] = df[TARGET].astype(float)
    # A handful of listings carry 0 for mileage or power; treat as missing.
    for col in ("mileage", "max_power"):
        df.loc[df[col] <= 0, col] = np.nan
    return df[["name"] + FEATURES + [TARGET]]


def load_v3(path: Path = V3_PATH, drop_duplicates: bool = True) -> pd.DataFrame:
    verify_checksum(path)
    return clean_v3(pd.read_csv(path), drop_duplicates=drop_duplicates)


def load_legacy(path: Path = LEGACY_PATH) -> pd.DataFrame:
    """The 301-row 'car data.csv' used by the original 2022 notebook."""
    verify_checksum(path)
    return pd.read_csv(path)
