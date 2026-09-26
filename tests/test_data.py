import numpy as np
import pandas as pd
import pytest

from carprice import data as D


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Maruti Swift Dzire VDI", ("Maruti", "Swift")),
        ("Land Rover Discovery Sport TD4 HSE", ("Land Rover", "Discovery")),
        ("Mercedes-Benz New C-Class C 220 CDI", ("Mercedes-Benz", "C-Class")),
        ("Ashok Leyland Stile LE", ("Ashok Leyland", "Stile")),
    ],
)
def test_split_name(name, expected):
    assert D.split_name(name) == expected


def test_clean_v3_parses_units_and_owner():
    raw = pd.DataFrame(
        {
            "name": ["Maruti Swift VDI", "Maruti Wagon R CNG"],
            "year": [2014, 2012],
            "selling_price": [450000, 250000],
            "km_driven": [145500, 60000],
            "fuel": ["Diesel", "CNG"],
            "seller_type": ["Individual", "Dealer"],
            "transmission": ["Manual", "Manual"],
            "owner": ["First Owner", "Fourth & Above Owner"],
            "mileage": ["23.4 kmpl", "26.6 km/kg"],
            "engine": ["1248 CC", None],
            "max_power": ["74 bhp", " bhp"],
            "torque": ["190Nm@ 2000rpm", None],
            "seats": [5.0, None],
        }
    )
    df = D.clean_v3(raw)
    assert list(df.columns) == ["name", *D.FEATURES, D.TARGET]
    assert df.loc[0, "mileage"] == 23.4 and df.loc[1, "mileage"] == 26.6
    assert df.loc[0, "engine"] == 1248 and np.isnan(df.loc[1, "engine"])
    assert np.isnan(df.loc[1, "max_power"])
    assert df["owner"].tolist() == [1.0, 4.0]
    assert "torque" not in df.columns


def test_clean_v3_drops_exact_duplicates():
    raw = pd.read_csv(D.V3_PATH)
    assert len(raw) == 8128
    df = D.load_v3()
    assert len(df) == 8128 - raw.duplicated().sum() == 6926
    assert df[D.TARGET].gt(0).all()


def test_checksums_match():
    for path in (D.V3_PATH, D.LEGACY_PATH):
        D.verify_checksum(path)
