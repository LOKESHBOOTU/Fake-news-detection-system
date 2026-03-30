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
    color: #162033;
    font-family: "Segoe UI", "Public Sans", Arial, sans-serif;
    background:
        radial-gradient(circle at top left, #fff3da 0%, transparent 28%),
        radial-gradient(circle at top right, #e7f1ff 0%, transparent 24%),
        linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%);
}

.app-shell {
    max-width: 1180px;
    margin: 0 auto;
}

.hero {
    padding: 32px;
    border: 1px solid #d6e0ea;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 16px 34px rgba(18, 32, 51, 0.07);
}

.eyebrow {
    margin: 0 0 8px 0;
    color: #132238;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: 0;
    text-transform: none;
}

.hero h1 {
    margin: 0 0 12px 0;
    font-size: 1.55rem;
    line-height: 1.3;
    color: #31445c;
    font-weight: 700;
}

.hero p {
    margin: 0;
    max-width: 100%;
    color: #3d4c5f;
    font-size: 1.08rem;
    line-height: 1.65;
}

.stat-card,
.panel-card {
    border: 1px solid #d6e0ea;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 10px 26px rgba(18, 32, 51, 0.05);
}

.stat-card {
    padding: 18px;
    text-align: center;
}

.stat-card h3 {
    margin: 0;
    color: #58708d;
    font-size: 0.9rem;
    font-weight: 700;
}

.stat-card .value {
    margin-top: 8px;
    color: #102033;
    font-size: 1.7rem;
    font-weight: 800;
}

.panel-card {
    padding: 18px 18px 10px 18px;
}

.section-title {
    margin: 0 0 10px 0;
    color: #11243a;
    font-size: 1.08rem;
    font-weight: 800;
}

.result-banner {
    padding: 16px 18px;
    border-radius: 18px;
    color: #102033;
    font-weight: 600;
    background: linear-gradient(135deg, #eef6ff 0%, #ffffff 100%);
    border: 1px solid #cfe0f2;
}

.probability-card {
    padding: 16px 18px;
    border-radius: 16px;
    border: 1px solid #d7e1eb;
    background: #ffffff;
    color: #111827;
}

.probability-card h4 {
    margin: 0 0 12px 0;
    color: #111827;
    font-size: 1rem;
    font-weight: 700;
}

.probability-row {
    margin-top: 12px;
}

.probability-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    color: #1f2937;
    font-size: 0.98rem;
    font-weight: 600;
}

.probability-track {
    width: 100%;
    height: 10px;
    border-radius: 999px;
    background: #e6eef8;
    overflow: hidden;
}

.probability-fill {
    height: 100%;
    border-radius: 999px;
    background: #3b82f6;
}

.gradio-container .block {
    border: none;
}

.gradio-container label,
.gradio-container .form label,
.gradio-container .wrap span,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose strong {
    color: #162033 !important;
    font-size: 0.98rem !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #ffffff !important;
    color: #162033 !important;
    border: 1px solid #cbd8e6 !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #728399 !important;
}

.gradio-container .form {
    background: #ffffff !important;
    border: 1px solid #d7e1eb !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 20px rgba(18, 32, 51, 0.04) !important;
}

.gradio-container .form > .wrap,
.gradio-container .form .wrap,
.gradio-container .form .block,
.gradio-container .form .block-content,
.gradio-container .form .gr-box,
.gradio-container .form .gr-panel {
    background: #ffffff !important;
}

.gradio-container button {
    font-weight: 700 !important;
    border-radius: 14px !important;
    font-size: 0.98rem !important;
}

.gradio-container button.primary {
    background: linear-gradient(135deg, #1b70f1 0%, #1558c0 100%) !important;
    color: #ffffff !important;
    border: none !important;
}

.gradio-container button.secondary {
    background: #e9eef5 !important;
    color: #17324f !important;
    border: 1px solid #cbd8e6 !important;
}

.gradio-container table,
.gradio-container thead,
.gradio-container tbody,
.gradio-container tr,
.gradio-container th,
.gradio-container td {
    color: #162033 !important;
    background: #ffffff !important;
    border-color: #dbe3ec !important;
    font-size: 0.98rem !important;
    line-height: 1.45 !important;
}

.gradio-container th {
    background: #eaf2fb !important;
    font-weight: 700 !important;
    color: #14314d !important;
}

.gradio-container thead th,
.gradio-container table thead th,
.gradio-container .dataframe-container thead th,
.gradio-container .dataframe-container th {
    background: #eef4fb !important;
    color: #0f2740 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}

.gradio-container thead th span,
.gradio-container table thead th span,
.gradio-container .dataframe-container thead th span,
.gradio-container .dataframe-container th span {
    color: #0f2740 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

.gradio-container .table-wrap,
.gradio-container .wrap .table-wrap {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid #dbe3ec !important;
}

.gradio-container .label-wrap,
.gradio-container .label,
.gradio-container .block-title {
    color: #11243a !important;
    font-weight: 700 !important;
}

.gradio-container .label-wrap,
.gradio-container .label-wrap > div,
.gradio-container .label-wrap > label,
.gradio-container .block-title,
.gradio-container label[data-testid="block-label"],
.gradio-container .form label,
.gradio-container .form label span,
.gradio-container .label-wrap span,
.gradio-container .label-wrap p,
.gradio-container .block-title span {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #111827 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    border-radius: 0 !important;
}

.gradio-container .dataframe-container .label,
.gradio-container .dataframe-container .label-wrap,
.gradio-container .dataframe-container .label-wrap *,
.gradio-container .dataframe-container [data-testid="block-label"],
.gradio-container .dataframe-container .block-title,
.gradio-container .dataframe-container .block-info {
    color: #1f2937 !important;
    background: transparent !important;
    opacity: 1 !important;
}

.gradio-container .examples table td {
    font-size: 0.97rem !important;
}

.gradio-container .examples table tr:hover td {
    background: #edf5ff !important;
}

.gradio-container .dataframe-container td {
    padding: 10px 12px !important;
}

.gradio-container .dataframe-container tr:nth-child(even) td {
    background: #f8fbff !important;
}

.gradio-container .dataframe-container tr:hover td {
    background: #eaf4ff !important;
}

.gradio-container .dataframe-container th {
    padding: 11px 12px !important;
}

.gradio-container .dataframe-container {
    font-family: "Segoe UI", "Public Sans", Arial, sans-serif !important;
}

.gradio-container .label-container,
.gradio-container .label-container *,
.gradio-container .label-container .wrap,
.gradio-container .label-container .block,
.gradio-container .label-container .block-content {
    background: #ffffff !important;
    color: #111827 !important;
}

.gradio-container .label-container {
    border: 1px solid #d7e1eb !important;
    border-radius: 16px !important;
    box-shadow: none !important;
}

.gradio-container .label-container .label,
.gradio-container .label-container .label-wrap,
.gradio-container .label-container [data-testid="block-label"] {
    background: transparent !important;
    color: #111827 !important;
}

.gradio-container .label-container .bar,
.gradio-container .label-container .progress-bar,
.gradio-container .label-container [role="progressbar"] {
    background: #3b82f6 !important;
    color: #111827 !important;
}

.gradio-container .label-container .value,
.gradio-container .label-container .name,
.gradio-container .label-container .confidence,
.gradio-container .label-container .score {
    color: #111827 !important;
    opacity: 1 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

.gradio-container .panel-card + div,
.gradio-container .stat-card + div {
    margin-top: 10px;
}
"""

EXAMPLES = [
    [
        "Health agency releases annual vaccine safety report with updated findings",
        "The national health agency released its annual vaccine safety report after reviewing hospital records, public health monitoring data, and independent advisory recommendations. Officials said the report includes updated findings on side effects, safety trends across age groups, and guidance for clinics preparing for the next immunization cycle. The document was published on the agency website along with supporting data tables and methodology notes for public review.",
    ],
    [
        "Secret machine controls weather in every city, viral blog claims",
        "A viral blog post claims that a hidden machine is secretly controlling weather patterns in major cities around the world. The article says the device can create storms, heat waves, and floods on command, but it does not provide scientific evidence, official records, or verified expert sources. The claim spreads through sensational language and conspiracy-style accusations rather than documented reporting.",
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


def _probability_summary(probabilities: np.ndarray) -> str:
    real_score = float(probabilities[0])
    fake_score = float(probabilities[1])
    return (
        "<div class='probability-card'>"
        "<h4>Class probabilities</h4>"
        "<div class='probability-row'>"
        f"<div class='probability-meta'><span>FAKE</span><span>{fake_score:.1%}</span></div>"
        f"<div class='probability-track'><div class='probability-fill' style='width:{fake_score * 100:.1f}%'></div></div>"
        "</div>"
        "<div class='probability-row'>"
        f"<div class='probability-meta'><span>REAL</span><span>{real_score:.1%}</span></div>"
        f"<div class='probability-track'><div class='probability-fill' style='width:{real_score * 100:.1f}%'></div></div>"
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
                _probability_summary(np.array([0.0, 0.0])),
            )

        model = models[selected_model]
        vector = vectorizer.transform([full_text])
        probabilities = _predict_probabilities(model, vector)
        label, confidence = _confidence_and_label(probabilities)
        detail = _result_summary(label, confidence, probabilities)
        explanation = _top_contributions(model, vector, feature_names)
        probability_html = _probability_summary(probabilities)
        return label, confidence, detail, explanation, probability_html

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Public Sans"),
    )

    with gr.Blocks(title="Fake News Detector", css=APP_CSS, theme=theme) as demo:
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
                probability_out = gr.HTML(label="Class probabilities")

        with gr.Row():
            with gr.Column(scale=4):
                gr.HTML("<div class='panel-card'><div class='section-title'>Why The Model Leaned This Way</div></div>")
                contrib_out = gr.Dataframe(
                    headers=["token", "contribution"],
                    wrap=True,
                )
            with gr.Column(scale=2):
                gr.HTML("<div class='panel-card'><div class='section-title'>Model Metrics</div></div>")
                gr.Dataframe(
                    value=metrics_table,
                    wrap=True,
                    column_widths=["150px", "92px", "92px", "92px", "70px"],
                )

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
                _probability_summary(np.array([0.0, 0.0])),
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
