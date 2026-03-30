from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_MODEL_NAME
from .data_utils import clean_text_keep_punct


MODEL_FILENAMES = {
    "LogisticRegression": "logisticregression.joblib",
    "NaiveBayes": "naivebayes.joblib",
    "SVM": "svm.joblib",
}


def _confidence_and_label(probabilities: np.ndarray) -> tuple[str, str]:
    fake_score = float(probabilities[1])
    label = "FAKE" if fake_score >= 0.5 else "REAL"
    confidence = max(fake_score, 1 - fake_score)
    return label, f"{confidence:.1%}"


def _softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _load_runtime(artifacts_dir: Path) -> tuple:
    metrics_path = artifacts_dir / "metrics.json"
    vectorizer_path = artifacts_dir / "tfidf.joblib"

    missing = [
        str(path.name)
        for path in [metrics_path, vectorizer_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts: {}. Run `python scripts/train.py` first.".format(
                ", ".join(missing)
            )
        )

    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    vectorizer = joblib.load(vectorizer_path)

    models = {}
    for model_name, filename in MODEL_FILENAMES.items():
        path = artifacts_dir / filename
        if path.exists():
            models[model_name] = joblib.load(path)

    if not models:
        raise FileNotFoundError(
            "No trained model files were found in the artifacts directory."
        )

    feature_names = np.array(vectorizer.get_feature_names_out())
    return vectorizer, models, feature_names, metadata


def _predict_probabilities(model, vector) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(vector)[0], dtype=float)

    decision = model.decision_function(vector)
    if np.ndim(decision) == 1:
        decision = np.array([-decision[0], decision[0]], dtype=float)
    return _softmax_from_scores(decision)


def _top_contributions(model, vector, feature_names: np.ndarray) -> pd.DataFrame:
    non_zero = vector.nonzero()[1]
    if len(non_zero) == 0:
        return pd.DataFrame(columns=["token", "contribution"])

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        contributions = [
            (feature_names[index], float(vector[0, index] * coefficients[index]))
            for index in non_zero
        ]
    elif hasattr(model, "feature_log_prob_"):
        log_prob = model.feature_log_prob_
        contributions = [
            (
                feature_names[index],
                float(vector[0, index] * (log_prob[1, index] - log_prob[0, index])),
            )
            for index in non_zero
        ]
    else:
        contributions = []

    contributions = sorted(contributions, key=lambda item: -abs(item[1]))[:15]
    return pd.DataFrame(contributions, columns=["token", "contribution"])


def build_demo(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> gr.Blocks:
    try:
        vectorizer, models, feature_names, metadata = _load_runtime(artifacts_dir)
    except FileNotFoundError as exc:
        with gr.Blocks() as error_demo:
            gr.Markdown("# Fake News Detector")
            gr.Markdown(f"**Setup required:** {exc}")
            gr.Markdown(
                "This app needs pre-trained artifacts. Place your dataset in `data/` and run `python scripts/train.py`."
            )
        return error_demo

    model_choices = list(models.keys())
    default_model = metadata.get("default_model", DEFAULT_MODEL_NAME)
    if default_model not in model_choices:
        default_model = model_choices[0]

    metrics_table = pd.DataFrame(metadata.get("models", {})).T.reset_index()
    metrics_table.columns = ["model", "accuracy", "precision", "recall", "f1"]

    def predict(selected_model: str, title: str, body: str):
        full_text = clean_text_keep_punct(f"{title or ''} {body or ''}".strip())
        if len(full_text) < 10:
            empty = pd.DataFrame(columns=["token", "contribution"])
            return (
                "Need more text",
                "Please enter at least 10 characters of title/body content.",
                "N/A",
                empty,
            )

        model = models[selected_model]
        vector = vectorizer.transform([full_text])
        probabilities = _predict_probabilities(model, vector)
        label, confidence = _confidence_and_label(probabilities)
        detail = f"REAL: {probabilities[0]:.3f} | FAKE: {probabilities[1]:.3f}"
        explanation = _top_contributions(model, vector, feature_names)
        return label, confidence, detail, explanation

    with gr.Blocks(title="Fake News Detector") as demo:
        gr.Markdown("# Fake News Detector")
        gr.Markdown(
            "A classical ML demo that scores an article as **REAL** or **FAKE** using TF-IDF features and a trained text classifier."
        )
        with gr.Row():
            with gr.Column(scale=2):
                title_in = gr.Textbox(label="Headline", placeholder="Optional headline")
                body_in = gr.Textbox(
                    label="Article text",
                    lines=12,
                    placeholder="Paste the article body or claim you want to analyze",
                )
                model_in = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Model",
                )
                submit = gr.Button("Analyze")
                label_out = gr.Textbox(label="Prediction")
                confidence_out = gr.Textbox(label="Confidence")
                detail_out = gr.Textbox(label="Scores")
                contrib_out = gr.Dataframe(
                    headers=["token", "contribution"],
                    label="Top contributing tokens",
                )
            with gr.Column(scale=1):
                dataset_rows = metadata.get("dataset", {}).get("rows_used", "unknown")
                best_model = metadata.get("best_model_by_f1", "unknown")
                gr.Markdown(
                    f"### Model Snapshot\n"
                    f"- Rows used: **{dataset_rows}**\n"
                    f"- Default model: **{default_model}**\n"
                    f"- Best by F1: **{best_model}**"
                )
                gr.Dataframe(value=metrics_table, label="Evaluation metrics")

        submit.click(
            predict,
            inputs=[model_in, title_in, body_in],
            outputs=[label_out, confidence_out, detail_out, contrib_out],
        )

    return demo
