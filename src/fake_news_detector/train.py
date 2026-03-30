from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_NAME,
    NGRAM_RANGE,
    RANDOM_STATE,
    SAMPLE_SIZE,
    TEST_SIZE,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
)
from .data_utils import load_dataset


REQUIRED_ARTIFACTS = [
    "tfidf.joblib",
    "logisticregression.joblib",
    "naivebayes.joblib",
    "svm.joblib",
    "metrics.json",
    "metrics.md",
]


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def artifacts_ready(artifacts_dir: Path) -> bool:
    return all((artifacts_dir / name).exists() for name in REQUIRED_ARTIFACTS)


def load_existing_metadata(artifacts_dir: Path) -> dict:
    return json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))


def train_and_save(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    sample_size: int | None = SAMPLE_SIZE,
    force_retrain: bool = False,
) -> dict:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if artifacts_ready(artifacts_dir) and not force_retrain:
        return load_existing_metadata(artifacts_dir)

    df, dataset_metadata = load_dataset(dataset_path=dataset_path, sample_size=sample_size)
    X_train, X_test, y_train, y_test = train_test_split(
        df["full_text"],
        df["label_mapped"],
        test_size=TEST_SIZE,
        stratify=df["label_mapped"],
        random_state=RANDOM_STATE,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=NGRAM_RANGE,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "NaiveBayes": MultinomialNB(),
        "SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    }

    metrics = {}
    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)
        metrics[model_name] = compute_metrics(y_test, predictions)
        joblib.dump(model, artifacts_dir / f"{model_name.lower()}.joblib")

    joblib.dump(vectorizer, artifacts_dir / "tfidf.joblib")

    best_model_name = max(metrics, key=lambda name: metrics[name]["f1"])
    metadata = {
        "default_model": best_model_name if best_model_name in models else DEFAULT_MODEL_NAME,
        "best_model_by_f1": best_model_name,
        "dataset": dataset_metadata,
        "models": metrics,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "vectorizer": {
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "ngram_range": list(NGRAM_RANGE),
        },
    }

    (artifacts_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Model Results",
        "",
        "| Model | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name, values in metrics.items():
        markdown_lines.append(
            "| {name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                name=model_name,
                **values,
            )
        )
    markdown_lines.append("")
    markdown_lines.append(f"Default deployed model: **{metadata['default_model']}**")
    (artifacts_dir / "metrics.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fake news detection models.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if saved artifacts already exist.",
    )
    args = parser.parse_args()
    metadata = train_and_save(force_retrain=args.force)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
