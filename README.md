# Fake News Detection using Machine Learning

## 📖 Description

This project detects whether a news article is **real** or **fake** by analyzing its textual content with machine learning models. It combines a reproducible training pipeline, saved model artifacts, and a Gradio interface for interactive prediction.

Fake news detection is an important real-world problem because misinformation spreads quickly through websites, social media platforms, and messaging apps. A system like this can help flag suspicious content and support users in identifying potentially misleading news.

## What Is Fake News Detection

Fake news detection is the process of identifying whether a news article or claim is trustworthy or misleading based on its content. In Natural Language Processing, this is treated as a text classification problem where the model learns patterns from labeled news data and predicts whether new input is likely to be **real** or **fake**.

## Live Demo

Try the deployed app on Hugging Face Spaces:

[Click here to test Fake News Detection Space](https://huggingface.co/spaces/Lokeshlokey/fake-news-detection-system)

## 🎯 Objectives

- Detect fake and real news from article text
- Compare multiple machine learning models
- Improve prediction quality using text preprocessing and TF-IDF features
- Provide an interactive interface for user input and prediction

## 🧠 Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Gradio
- Joblib

## 📂 Dataset Information

- Dataset used: `WELFake_Dataset.csv`
- Total records in the dataset file: **362,555**
- Main features:
  - `title`
  - `text`
  - `label`
- Label format:
  - `0` = Real news
  - `1` = Fake news

For faster experimentation and deployment, the current training pipeline uses a cleaned sampled subset. The latest saved training run used **19,979** rows.

## Project Structure

```text
Fake News Detection system/
|-- app.py
|-- artifacts/
|-- data/
|-- DSP_PBL_PROJECT.ipynb
|-- README.md
|-- requirements.txt
|-- scripts/
|   `-- train.py
`-- src/
    `-- fake_news_detector/
```

## Requirements

- Python 3.10 or later
- pip

## ⚙️ Installation & Setup

```bash
git clone https://github.com/LOKESHBOOTU/Fake-news-detection-system.git
cd Fake-news-detection-system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Run Locally

After installing the dependencies, start the app with:

```bash
python app.py
```

Then open the local Gradio interface in your browser:

[http://127.0.0.1:7860](http://127.0.0.1:7860)

## 🔍 Methodology / Workflow

1. **Data Collection**  
   The project uses the WELFake dataset containing article title, article text, and labels.

2. **Data Cleaning**  
   Missing values and empty rows are removed before training.

3. **Text Preprocessing**  
   Text is converted to lowercase, links and emails are removed, and noisy characters are cleaned.

4. **Feature Extraction**  
   TF-IDF vectorization is applied using unigram and bigram features.

5. **Model Training**  
   Multiple classical machine learning models are trained on the processed text.

6. **Model Evaluation**  
   Accuracy, precision, recall, and F1-score are used to compare performance.

7. **Prediction**  
   The saved best-performing setup is loaded into the Gradio app for interactive prediction.

## How It Works

1. Enter a news headline and article text in the app.
2. The input text is cleaned and normalized.
3. TF-IDF converts the text into numerical features.
4. A trained machine learning model predicts whether the input is real or fake.
5. The app displays the prediction, confidence score, class probabilities, and contributing tokens.

## 🤖 Machine Learning Models Used

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

Note: the current implementation does **not** include Random Forest yet.

## 📊 Results / Accuracy

Latest saved model metrics:

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9297 | 0.9264 | 0.9377 | 0.9320 |
| Naive Bayes | 0.8799 | 0.9103 | 0.8500 | 0.8792 |
| SVM | 0.9474 | 0.9471 | 0.9508 | 0.9490 |

**Best-performing model:** SVM

## 📸 Screenshots / Output

The project includes a Gradio interface for entering a headline and article text, viewing predictions, and inspecting token contribution details.

### Main Interface

<img width="1623" height="990" alt="Main interface" src="https://github.com/user-attachments/assets/0f11df78-82ed-41ed-86f9-a8035118bc68" />

### Model Metrics And Token Contribution View

<img width="1563" height="542" alt="Metrics and contribution view" src="https://github.com/user-attachments/assets/3ebe48c3-97a8-4dd5-98b7-2eed93f49bb6" />

## 🚀 Features

- Detect fake news instantly from user input
- Interactive web interface for prediction
- Comparison across multiple machine learning models
- Saved trained artifacts so the app does not retrain on every run
- Deployment-ready project structure for local use and Hugging Face Spaces

## Applications

- Identifying misleading news headlines and articles
- Demonstrating NLP text classification in academic projects
- Comparing classical machine learning models on text data
- Building portfolio-ready machine learning web apps

## Why This Project Is Useful

- Helps users understand how fake news classification works in practice
- Saves time by giving a quick prediction on suspicious content
- Shows a complete workflow from training to deployment
- Useful for students, researchers, and beginners learning NLP and ML deployment

## ⚠️ Limitations

- Performance depends heavily on dataset quality
- The model may struggle with unseen writing styles or new misinformation patterns
- It is a machine learning classifier, not a complete fact-checking system
- The current app is optimized for demonstration and portfolio use rather than large-scale production

## 🔮 Future Improvements

- Add deep learning models such as LSTM or BERT
- Include more diverse and up-to-date datasets
- Improve prediction explanations and confidence display
- Add stronger production deployment polish beyond the current Hugging Face Space
- Add more model comparison options such as Random Forest or XGBoost

## 👨‍💻 Author / Contributors

- **Lokesh Bootu**
- GitHub: [LOKESHBOOTU](https://github.com/LOKESHBOOTU)

## 📄 License

No license file has been added yet.

If you want to open-source this project properly, adding an **MIT License** would be a good next step.

## Deployment

This project is deployed on Hugging Face Spaces and can also be run locally using the saved model artifacts.
