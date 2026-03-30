# Fake News Detection using Machine Learning

## 📖 Description

This project detects whether a news article is **real** or **fake** by analyzing its textual content using machine learning models. It is built as a small end-to-end application with a training pipeline, saved model artifacts, and a Gradio interface for interactive prediction.

Fake news detection is an important real-world problem because misinformation spreads quickly through websites, social media, and messaging platforms. A system like this can help flag suspicious content and support users in identifying potentially misleading news.

## 🎯 Objectives

- Detect fake vs real news from article text
- Compare multiple machine learning models
- Improve prediction quality using text preprocessing and TF-IDF features
- Provide a simple interface for testing predictions interactively

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
- Total records in dataset file: **362,555**
- Main features used:
  - `title`
  - `text`
  - `label`
- Label meaning:
  - `0` = Real news
  - `1` = Fake news

This project currently trains on a cleaned sampled subset for faster experimentation and app deployment. The latest saved training run used **19,979** rows.

## ⚙️ Installation & Setup

```bash
git clone https://github.com/LOKESHBOOTU/Fake-news-detection-system.git
cd Fake-news-detection-system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 🔍 Methodology / Workflow

1. Data Collection  
   The project uses the WELFake dataset containing article title, article text, and label.

2. Data Cleaning  
   Missing values and empty rows are handled before training.

3. Text Preprocessing  
   Text is converted to lowercase, links and emails are removed, and noisy characters are cleaned.

4. Feature Extraction  
   TF-IDF vectorization is used with unigram and bigram features.

5. Model Training  
   Multiple classical machine learning models are trained on the processed text.

6. Model Evaluation  
   Accuracy, precision, recall, and F1-score are calculated.

7. Prediction  
   The saved model is loaded in the Gradio app to classify new user input.

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

The project includes a Gradio interface for entering a headline and article text, viewing predictions, and inspecting model contribution details.

Add your screenshots to the `assets/` folder with these names:

- `assets/ui-main.png`
- `assets/ui-metrics.png`

Then they will appear here automatically:

### Main Interface

![Main Interface](assets/ui-main.png)

### Model Metrics And Token Contribution View

![Metrics And Contribution View](assets/ui-metrics.png)

## 🚀 Features

- Detect fake news instantly from user input
- Interactive prediction using a web interface
- Multiple machine learning model comparison
- Saved trained artifacts so the app does not retrain every time
- Clean project structure for local use and deployment

## ⚠️ Limitations

- Performance depends heavily on dataset quality
- The model may fail on unseen writing styles or new misinformation patterns
- It is a machine learning classifier, not a full fact-checking system
- Current deployment is optimized for demo use rather than large-scale production

## 🔮 Future Improvements

- Add deep learning models such as LSTM or BERT
- Include more diverse and up-to-date datasets
- Improve UI design and prediction explanations
- Deploy the app publicly on Hugging Face Spaces
- Add more model comparison options such as Random Forest or XGBoost

## 👨‍💻 Author / Contributors

- **Lokesh Bootu**
- GitHub: [LOKESHBOOTU](https://github.com/LOKESHBOOTU)

## 📄 License

No license file has been added yet.

If you want to open-source this project properly, adding an **MIT License** would be a good next step.


