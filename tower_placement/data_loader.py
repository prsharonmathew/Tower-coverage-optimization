from __future__ import annotations

from pathlib import Path

import pandas as pd


class CityDataLoader:
    def load(self, file_path: Path) -> pd.DataFrame:
        raw_df = pd.read_csv(
            file_path,
            sep=None,
            engine="python",
            encoding="utf-8",
            skipinitialspace=True,
            header=None,
            names=["city", "latitude", "longitude"],
        )

        city_ids = raw_df["city"].astype(str).str.strip()
        city_ids = city_ids.where(city_ids != "", raw_df.index.astype(str))

        cities = pd.DataFrame(
            {
                "city_id": city_ids,
                "latitude": pd.to_numeric(raw_df["latitude"], errors="coerce"),
                "longitude": pd.to_numeric(raw_df["longitude"], errors="coerce"),
            }
        ).reset_index(drop=True)

        invalid_rows = cities[cities["latitude"].isna() | cities["longitude"].isna()]
        if not invalid_rows.empty:
            raise ValueError(
                "Found invalid latitude/longitude values in the input data. "
                f"{invalid_rows.index.tolist()}"
            )

        return cities
