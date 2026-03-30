from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TEXT_CANDIDATES = ("title", "text", "article", "content", "body", "news")
LABEL_CANDIDATES = ("label", "target", "truth", "class", "is_fake", "fake")


def clean_text_keep_punct(value: str | None) -> str:
    """Normalize input text while preserving common sentence punctuation."""
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\.,?!\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_text_and_label_columns(sample_df: pd.DataFrame) -> tuple[list[str], str]:
    columns = list(sample_df.columns)
    lower_map = {col.lower(): col for col in columns}

    if "title" in lower_map and "text" in lower_map:
        text_columns = [lower_map["title"], lower_map["text"]]
    else:
        text_columns = [
            col for col in columns if any(token in col.lower() for token in TEXT_CANDIDATES)
        ]
        if not text_columns:
            object_columns = sample_df.select_dtypes(include=["object"]).columns.tolist()
            if object_columns:
                text_columns = object_columns[:1]
            else:
                raise ValueError("Could not detect a text column in the dataset.")
        else:
            text_columns = text_columns[:2]

    label_columns = [
        col for col in columns if any(token in col.lower() for token in LABEL_CANDIDATES)
    ]
    if not label_columns:
        raise ValueError(
            "Could not detect a label column. Use a column like 'label', 'class', or 'is_fake'."
        )

    return text_columns, label_columns[0]


def map_label_safe(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    unique_values = normalized.dropna().unique().tolist()
    if len(unique_values) != 2:
        raise ValueError(
            f"Expected a binary label column but found {len(unique_values)} unique values: {unique_values[:10]}"
        )

    truthy_fake = {"1", "fake", "false", "f", "yes", "y"}
    truthy_real = {"0", "real", "true", "t", "no", "n"}

    if set(unique_values).issubset(truthy_fake | truthy_real):
        mapping = {}
        for value in unique_values:
            if value in truthy_fake:
                mapping[value] = 1
            elif value in truthy_real:
                mapping[value] = 0
        return normalized.map(mapping).astype(int)

    if any("fake" in value for value in unique_values) or any("real" in value for value in unique_values):
        return normalized.map(lambda value: 1 if "fake" in value else 0).astype(int)

    try:
        numeric = normalized.astype(int)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported label values: {unique_values[:10]}. Use binary real/fake or 0/1 labels."
        ) from exc

    numeric_values = set(numeric.unique().tolist())
    if numeric_values != {0, 1}:
        raise ValueError(
            f"Numeric labels must be 0/1, but found {sorted(numeric_values)}."
        )
    return numeric


def load_dataset(dataset_path: Path, sample_size: int | None = None) -> tuple[pd.DataFrame, dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Put the CSV in the data/ folder or set FAKE_NEWS_DATASET."
        )

    sample_df = pd.read_csv(dataset_path, nrows=10)
    text_columns, label_column = detect_text_and_label_columns(sample_df)
    available_columns = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    read_columns = [col for col in [*text_columns, label_column] if col in available_columns]
    df = pd.read_csv(dataset_path, usecols=read_columns)

    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

    df["full_text_raw"] = df[text_columns].fillna("").agg(" ".join, axis=1)
    df = df[df["full_text_raw"].str.strip() != ""].dropna(subset=[label_column]).reset_index(drop=True)
    df["label_mapped"] = map_label_safe(df[label_column])

    if sample_size and len(df) > sample_size:
        sampled_indices = (
            df.groupby("label_mapped", group_keys=False)
            .sample(frac=min(1, sample_size / len(df)), random_state=42)
            .index
        )
        sampled = df.loc[sampled_indices]
        if len(sampled) > sample_size:
            sampled = sampled.sample(n=sample_size, random_state=42)
        elif len(sampled) < sample_size:
            remainder = df.drop(index=sampled.index, errors="ignore")
            extra_needed = min(sample_size - len(sampled), len(remainder))
            if extra_needed:
                sampled = pd.concat(
                    [sampled, remainder.sample(n=extra_needed, random_state=42)],
                    ignore_index=True,
                )
        df = sampled.reset_index(drop=True)

    df["full_text"] = df["full_text_raw"].map(clean_text_keep_punct)
    df = df[df["full_text"].str.len() > 20].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows remained after cleaning. Check the dataset content.")

    metadata = {
        "dataset_path": str(dataset_path),
        "text_columns": text_columns,
        "label_column": label_column,
        "rows_used": int(len(df)),
        "label_distribution": {
            str(label): int(count)
            for label, count in df["label_mapped"].value_counts().sort_index().items()
        },
    }
    return df, metadata
