# Fake News Classification with Machine Learning

A short end-to-end NLP classification project that predicts whether a news article is **Fake** or **Real**. A short NLP project built with TF-IDF, Multinomial Naive Bayes, Logistic Regression, and a Streamlit web app.

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

