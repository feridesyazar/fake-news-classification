# Fake News Classification with Machine Learning

A short end-to-end NLP classification project that predicts whether a news article is **Fake** or **Real**.

## Problem Type

**Binary Classification / Natural Language Processing (NLP)**

- `0` = Fake
- `1` = Real

## Dataset

This project uses the **WELFake Dataset**, which contains 72,134 news articles in the original release.

Main columns:

- `title` - news headline
- `text` - article content
- `label` - target variable

Dataset sources:

- Kaggle: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
- Original Zenodo record: https://zenodo.org/records/4561253
- Paper: P. K. Verma et al., *WELFake: Word Embedding Over Linguistic Features for Fake News Detection*, IEEE Transactions on Computational Social Systems (2021), https://doi.org/10.1109/TCSS.2021.3068519

## Models

The notebook compares:

1. **Multinomial Naive Bayes** - a simple and fast text-classification baseline.
2. **Logistic Regression** - a strong classical model for sparse TF-IDF text features.

The best model is selected using the **weighted F1 score**.

## Workflow

1. Load and inspect the dataset
2. Handle missing text values
3. Remove duplicate articles
4. Combine title and article text
5. Visualize class distribution
6. Create an 80/20 train-test split
7. Convert text to TF-IDF features
8. Train Naive Bayes and Logistic Regression
9. Compare Accuracy and F1 Score
10. Evaluate with a Classification Report and Confusion Matrix
11. Make a sample prediction

## Project Structure

```text
fake_news_classification_project/
│
├── Fake_News_Classification.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── WELFake_Dataset.csv   # local only - do not upload to GitHub
```

## Installation

```bash
pip install -r requirements.txt
```

Then open:

```bash
jupyter notebook Fake_News_Classification.ipynb
```

## Dataset Setup

Download `WELFake_Dataset.csv` and place it inside the `data/` folder.

The dataset is large and should normally **not be committed to GitHub**. The `.gitignore` file in this project excludes CSV files inside `data/`.

## Runtime Note

The notebook uses a sample of **20,000 cleaned rows** by default so it can run quickly on a normal computer.

To use all cleaned rows, change:

```python
SAMPLE_SIZE = 20000
```

to:

```python
SAMPLE_SIZE = None
```

## Limitation

This is an educational machine learning project. The model detects patterns learned from the dataset; it does **not** independently verify facts and should not be used as a professional fact-checking system.

## Reference / Learning Inspiration

Aman Kharwal's introductory fake-news classification project is also a useful simple reference:

https://thecleverprogrammer.com/2021/06/30/fake-news-detection-with-machine-learning/
