# Fake News Detector

A machine learning project that classifies news articles as `REAL` or `FAKE` using classical NLP models built with TF-IDF features and scikit-learn.

This repository started as a notebook-based prototype and is now organized as a small app project with:

- a reusable training pipeline
- a standalone Gradio web app
- saved model artifacts for deployment
- documentation for local setup and hosting

## Overview

The project trains multiple text classification models on a labeled fake-news dataset and serves predictions through a simple interface where a user can enter a headline and article body.

The current implementation includes:

- Logistic Regression
- Naive Bayes
- Linear SVM

The default deployed model is Logistic Regression.

## Project Structure

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

- Reproducible training workflow with fixed configuration
- Automatic dataset column detection for text and labels
- Saved TF-IDF vectorizer and trained models
- Evaluation metrics exported after training
- Gradio interface for interactive predictions
- Hugging Face Spaces compatible app entrypoint

## Dataset Format

The training script supports automatic column detection, but the preferred CSV format is:

| Column | Description |
| --- | --- |
| `title` | News headline |
| `text` | Article content |
| `label` | Binary label such as `0/1` or `real/fake` |

Place your dataset at `data/WELFake_Dataset.csv`, or set the `FAKE_NEWS_DATASET` environment variable to point to another CSV file.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python scripts/train.py
```

This generates files inside `artifacts/` such as:

- `tfidf.joblib`
- `logisticregression.joblib`
- `naivebayes.joblib`
- `svm.joblib`
- `metrics.json`
- `metrics.md`

## Running the App

Start the Gradio application locally:

```bash
python app.py
```

Then open the local URL shown in the terminal.

If trained artifacts are missing, the app will show a setup message instead of failing silently.

## Deployment

This project is structured for deployment on Hugging Face Spaces.

### Hugging Face Spaces

1. Create a new Gradio Space.
2. Upload this repository or connect the GitHub repository.
3. Make sure the trained files exist in `artifacts/` before deployment.
4. Hugging Face will use `app.py` and `requirements.txt` automatically.

## GitHub

To push new local changes:

```bash
git add .
git commit -m "Update project"
git push
```

## Notes

- The dataset is not committed by default.
- The notebook `DSP_PBL_PROJECT.ipynb` is kept as the original research/prototype file.
- The deployed app is intended for demonstration and portfolio use rather than full production use.
