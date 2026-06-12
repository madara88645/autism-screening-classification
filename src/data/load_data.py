from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATH, DROP_COLUMNS, TARGET_COLUMN


NUMERIC_COLUMNS = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "age",
    "result",
]


def _clean_text_value(value):
    if not isinstance(value, str):
        return value

    cleaned = value.strip().strip("'").strip()
    return cleaned if cleaned else np.nan


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace("?", np.nan)

    object_columns = cleaned.select_dtypes(include="object").columns
    for column in object_columns:
        cleaned[column] = cleaned[column].apply(_clean_text_value)

    for column in DROP_COLUMNS:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=column)

    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"NO": 0, "YES": 1}).astype(int)

    return cleaned


def load_dataset(data_path: Path | str = DATA_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(data_path)
    return clean_raw_dataframe(dataframe)

