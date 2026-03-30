from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_MODEL_NAME
from .data_utils import clean_text_keep_punct

APP_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, #f4efe3 0%, transparent 35%),
        radial-gradient(circle at top right, #dce9f7 0%, transparent 28%),
        linear-gradient(180deg, #fbfaf5 0%, #f2f4f7 100%);
}

.app-shell {
    max-width: 1180px;
    margin: 0 auto;
}

.hero {
    padding: 28px;
    border: 1px solid #d7dce2;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.86);
    box-shadow: 0 18px 40px rgba(34, 53, 74, 0.08);
}

.eyebrow {
    display: inline-block;
    margin-bottom: 12px;
    padding: 6px 12px;
    border-radius: 999px;
    background: #16253a;
    color: #f7f3ea;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 0 0 8px 0;
    font-size: 2.6rem;
    line-height: 1.05;
}

.hero p {
    margin: 0;
    max-width: 760px;
    color: #415063;
    font-size: 1.05rem;
}

.stat-card,
.panel-card {
    border: 1px solid #d7dce2;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 14px 30px rgba(34, 53, 74, 0.06);
}

.stat-card {
    padding: 18px;
    text-align: center;
}

.stat-card h3 {
    margin: 0;
    color: #5d6a79;
    font-size: 0.9rem;
    font-weight: 600;
}

.stat-card .value {
    margin-top: 8px;
    color: #122033;
    font-size: 1.7rem;
    font-weight: 700;
}

.panel-card {
    padding: 18px 18px 10px 18px;
}

.section-title {
    margin: 0 0 10px 0;
    color: #122033;
    font-size: 1.05rem;
    font-weight: 700;
}

.result-banner {
    padding: 16px 18px;
    border-radius: 18px;
    color: #102033;
    font-weight: 600;
    background: linear-gradient(135deg, #eef3f9 0%, #ffffff 100%);
    border: 1px solid #d7dce2;
}
"""

EXAMPLES = [
    [
        "Health agency releases annual vaccine safety report",
        "The agency published updated vaccine safety findings with methods, data tables, and recommendations for public review.",
    ],
    [
        "Secret machine controls weather in every city",
        "A viral blog claims a hidden device changes the weather worldwide overnight without providing evidence or official sources.",
    ],
]


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


def _result_summary(label: str, confidence: str, probabilities: np.ndarray) -> str:
    tone = "Likely fabricated or misleading." if label == "FAKE" else "Likely grounded in legitimate reporting."
    return (
        "<div class='result-banner'>"
        f"<div><strong>{label}</strong> with <strong>{confidence}</strong> confidence</div>"
        f"<div style='margin-top:6px; color:#4b5a6d; font-weight:500;'>{tone}</div>"
        f"<div style='margin-top:8px; color:#697789; font-size:0.94rem;'>"
        f"REAL score: {probabilities[0]:.3f} | FAKE score: {probabilities[1]:.3f}"
        "</div>"
        "</div>"
    )


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
    metrics_table[["accuracy", "precision", "recall", "f1"]] = metrics_table[
        ["accuracy", "precision", "recall", "f1"]
    ].round(4)

    dataset_rows = metadata.get("dataset", {}).get("rows_used", "unknown")
    best_model = metadata.get("best_model_by_f1", "unknown")
    label_distribution = metadata.get("dataset", {}).get("label_distribution", {})
    real_count = label_distribution.get("0", "unknown")
    fake_count = label_distribution.get("1", "unknown")

    def predict(selected_model: str, title: str, body: str):
        full_text = clean_text_keep_punct(f"{title or ''} {body or ''}".strip())
        if len(full_text) < 10:
            empty = pd.DataFrame(columns=["token", "contribution"])
            return (
                "Need more text",
                "Please enter at least 10 characters of title/body content.",
                "<div class='result-banner'>Please enter at least 10 characters of title/body content.</div>",
                empty,
                {"REAL": 0.0, "FAKE": 0.0},
            )

        model = models[selected_model]
        vector = vectorizer.transform([full_text])
        probabilities = _predict_probabilities(model, vector)
        label, confidence = _confidence_and_label(probabilities)
        detail = _result_summary(label, confidence, probabilities)
        explanation = _top_contributions(model, vector, feature_names)
        chart = {"REAL": float(probabilities[0]), "FAKE": float(probabilities[1])}
        return label, confidence, detail, explanation, chart

    with gr.Blocks(title="Fake News Detector", css=APP_CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div class="app-shell">
              <div class="hero">
                <div class="eyebrow">Fake News Detection</div>
                <h1>Check whether a news claim looks real or suspicious</h1>
                <p>
                  Paste a headline and article body to compare how the trained models score the text.
                  The app uses classical NLP features, not a large language model, so the output should be treated as a demo signal rather than final fact-checking.
                </p>
              </div>
            </div>
            """
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML(
                    f"""
                    <div class="stat-card">
                      <h3>Rows Used</h3>
                      <div class="value">{dataset_rows}</div>
                    </div>
                    """
                )
            with gr.Column(scale=1):
                gr.HTML(
                    f"""
                    <div class="stat-card">
                      <h3>Default Model</h3>
                      <div class="value" style="font-size:1.2rem;">{default_model}</div>
                    </div>
                    """
                )
            with gr.Column(scale=1):
                gr.HTML(
                    f"""
                    <div class="stat-card">
                      <h3>Best By F1</h3>
                      <div class="value" style="font-size:1.2rem;">{best_model}</div>
                    </div>
                    """
                )
            with gr.Column(scale=1):
                gr.HTML(
                    f"""
                    <div class="stat-card">
                      <h3>Label Split</h3>
                      <div class="value" style="font-size:1.1rem;">R {real_count} / F {fake_count}</div>
                    </div>
                    """
                )

        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<div class='panel-card'><div class='section-title'>Article Input</div></div>")
                title_in = gr.Textbox(
                    label="Headline",
                    placeholder="Optional headline",
                    lines=2,
                )
                body_in = gr.Textbox(
                    label="Article text",
                    lines=12,
                    placeholder="Paste the article body or claim you want to analyze",
                )
                model_in = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="Model",
                    info="Use the default model for the most stable demo result.",
                )
                with gr.Row():
                    submit = gr.Button("Analyze Article", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[title_in, body_in],
                    label="Quick examples",
                )
            with gr.Column(scale=1):
                gr.HTML("<div class='panel-card'><div class='section-title'>Prediction Summary</div></div>")
                label_out = gr.Textbox(label="Prediction")
                confidence_out = gr.Textbox(label="Confidence")
                detail_out = gr.HTML(label="Summary")
                probability_out = gr.Label(label="Class probabilities")

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML("<div class='panel-card'><div class='section-title'>Why The Model Leaned This Way</div></div>")
                contrib_out = gr.Dataframe(
                    headers=["token", "contribution"],
                    label="Top contributing tokens",
                )
            with gr.Column(scale=1):
                gr.HTML("<div class='panel-card'><div class='section-title'>Model Metrics</div></div>")
                gr.Dataframe(value=metrics_table, label="Evaluation metrics", wrap=True)

        submit.click(
            predict,
            inputs=[model_in, title_in, body_in],
            outputs=[label_out, confidence_out, detail_out, contrib_out, probability_out],
        )
        clear.click(
            lambda: (
                "",
                "",
                default_model,
                "",
                "",
                "<div class='result-banner'>Run an analysis to see the result summary here.</div>",
                pd.DataFrame(columns=["token", "contribution"]),
                {"REAL": 0.0, "FAKE": 0.0},
            ),
            outputs=[
                title_in,
                body_in,
                model_in,
                label_out,
                confidence_out,
                detail_out,
                contrib_out,
                probability_out,
            ],
        )

    return demo
