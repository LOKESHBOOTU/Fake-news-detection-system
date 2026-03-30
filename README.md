# Fake News Detector

A portfolio-ready machine learning demo that classifies article text as `REAL` or `FAKE` using classical NLP models built on TF-IDF features.

The original project started as a single Colab-style notebook. This repo keeps that notebook for reference and adds a clean training script, a standalone Gradio app, and project files that are ready for GitHub and Hugging Face Spaces.

## Project layout

```text
.
|-- app.py
|-- artifacts/
|-- data/
|-- DSP_PBL_PROJECT.ipynb
|-- requirements.txt
|-- scripts/
|   `-- train.py
`-- src/
    `-- fake_news_detector/
```

## Features

- Standalone training workflow with reproducible settings
- Automatic text/label column detection for compatible CSV datasets
- Three classical models: Logistic Regression, Naive Bayes, and Linear SVM
- Saved model artifacts and metrics for deployment
- Gradio interface for testing article headlines and body text
- Hugging Face Spaces friendly `app.py` entrypoint

## Preferred dataset schema

The training script can auto-detect columns, but the cleanest CSV format is:

| Column | Meaning |
| --- | --- |
| `title` | Article headline |
| `text` | Article body |
| `label` | Binary label where `0=REAL` and `1=FAKE`, or values like `real` / `fake` |

Place your dataset at [data/WELFake_Dataset.csv](C:\Users\LOKESHBOOTU\OneDrive\Desktop\Fake News Detection system\data\WELFake_Dataset.csv), or set the `FAKE_NEWS_DATASET` environment variable to a different CSV path.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the models

```bash
python scripts/train.py
```

Training creates these local files in `artifacts/`:

- `tfidf.joblib`
- `logisticregression.joblib`
- `naivebayes.joblib`
- `svm.joblib`
- `metrics.json`
- `metrics.md`

## Run the app locally

```bash
python app.py
```

If artifacts are missing, the app shows a clear setup message instead of relying on notebook state.

## Hugging Face Spaces deployment

This project is structured for a Gradio Space:

1. Create a new Gradio Space on Hugging Face.
2. Push this repository to GitHub.
3. Upload or sync the repo contents into the Space.
4. Make sure the trained artifacts exist in `artifacts/` before deployment, or generate them and include only the small artifact files you want to host.
5. Hugging Face will detect `app.py` and `requirements.txt` automatically.

## GitHub publishing checklist

```bash
git init
git branch -M main
git add .
git commit -m "Initial fake news detector app"
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Notes

- The dataset is intentionally not committed by default.
- The default deployed model is Logistic Regression, while the app still allows comparison with Naive Bayes and SVM when their artifacts are present.
- The original notebook, [DSP_PBL_PROJECT.ipynb](C:\Users\LOKESHBOOTU\OneDrive\Desktop\Fake News Detection system\DSP_PBL_PROJECT.ipynb), remains in the repo as a research reference.
