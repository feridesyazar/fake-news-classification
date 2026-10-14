# 📰 Fake News Classification

**An End-to-End Natural Language Processing (NLP) and Machine Learning Project**

This project builds a machine learning model to classify news articles as **Fake** or **Real** using Natural Language Processing (NLP).

The project covers data preprocessing, exploratory data analysis, model training, evaluation, and deployment through an interactive **Streamlit web application**.

## 🎯 Project Objective

The goal is to develop a simple and practical fake news classification system that:

- Analyzes news headlines and article content.
- Converts text into numerical features using **TF-IDF**.
- Compares two classification algorithms.
- Evaluates model performance.
- Provides predictions through a Streamlit web application.

## 📊 Dataset

The project uses the **WELFake Dataset**, containing **72,134 news articles** in the original release.

| Feature | Description |
|---|---|
| `title` | News headline |
| `text` | News article content |
| `label` | Target variable |

**Target Labels:**
- `0` — Fake News
- `1` — Real News

**Dataset Sources:**

- [Kaggle – WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- [Zenodo – Original Dataset](https://zenodo.org/records/4561253)

## 🔍 Exploratory Data Analysis

The dataset is inspected and prepared before model training.

Main steps:

1. Check dataset structure and missing values.
2. Remove duplicate news articles.
3. Combine headlines and article text.
4. Explore the Fake and Real class distribution.
5. Split the data into training and test sets.

## 🤖 Machine Learning Models

Two classification algorithms are compared:

| Model | Description |
|---|---|
| **Multinomial Naive Bayes** | A fast, probability-based baseline for text classification |
| **Logistic Regression** | A linear classification model suitable for sparse TF-IDF features |

**Text Vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency)

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The best-performing model is selected using the weighted F1 Score.

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- TF-IDF
- Joblib
- Jupyter Notebook
- Streamlit

## 🌐 Streamlit Web Application

An interactive Streamlit application allows users to classify news using three input methods:

1. **Enter News Manually** — Type or paste a news headline and article.
2. **Paste News URL** — Extract available text from a news website.
3. **Upload News File** — Upload a `.txt` file containing news text.

The application uses the trained machine learning model to predict whether the submitted content resembles **Fake News** or **Real News**.

**Live Demo:** Coming soon

*Note: The model identifies learned text patterns and does not independently verify the factual accuracy of news articles.*

## 📁 Project Structure

```text
fake-news-classification/
│
├── Fake_News_Classification.ipynb
├── Fake_News_Classification_Streamlit.ipynb
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/
│   └── WELFake_Dataset.csv

```

*The CSV dataset is stored locally and is not included in the GitHub repository.*

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/feridesyazar/fake-news-classification.git
cd fake-news-classification
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit application**

```bash
python -m streamlit run app.py
```

**4. Open the application**

```text
http://localhost:8501
```

To explore the machine learning workflow, open the Jupyter notebooks.

## ✅ Conclusion

This project demonstrates a complete NLP classification workflow, from preparing text data to training and comparing machine learning models.

**Logistic Regression** and **Multinomial Naive Bayes** provide practical baseline approaches for fake news classification.

The Streamlit application extends the project beyond a notebook and allows users to interact with the trained model through a web interface.

## 📚 References

- [WELFake Dataset – Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- [WELFake Dataset – Zenodo](https://zenodo.org/records/4561253)
- Verma et al. (2021), *WELFake: Word Embedding Over Linguistic Features for Fake News Detection*, IEEE Transactions on Computational Social Systems. [DOI](https://doi.org/10.1109/TCSS.2021.3068519)
- [Aman Kharwal – Fake News Detection with Machine Learning](https://amanxai.com/2021/06/30/fake-news-detection-with-machine-learning/)

---

**Project Type:** Binary Classification | NLP | Machine Learning

**Status:** Model trained · Streamlit application developed · Cloud deployment in progress
