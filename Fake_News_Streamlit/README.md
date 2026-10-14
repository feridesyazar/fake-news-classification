# Fake News Classification

A short NLP project built with **TF-IDF**, **Multinomial Naive Bayes**, **Logistic Regression**, and a **Streamlit** web app.

## Project structure

```text
Fake_News_Classification_Complete/
├── Fake_News_Classification.ipynb            # Original ML notebook
├── Fake_News_Classification_Streamlit.ipynb  # Same workflow + save model
├── app.py                                    # Local Streamlit demo
├── requirements.txt
├── data/
│   └── WELFake_Dataset.csv                   # Download separately
├── models/                                   # Created by second notebook
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
└── huggingface_space/                        # Deployment-ready template
```

## Getting started (VS Code)

1. Open this project folder in VS Code.
2. Install the **Python** and **Jupyter** extensions.
3. Download the WELFake CSV and place it in `data/WELFake_Dataset.csv`.
4. In a terminal, run `pip install -r requirements.txt`.
5. Run either notebook with **Run All**. The Streamlit notebook saves model files in `models/`.
6. Run `streamlit run app.py`, then open the local URL in your browser.

## Two notebook versions

- **Fake_News_Classification.ipynb** is the original standalone ML project.
- **Fake_News_Classification_Streamlit.ipynb** contains the same ML workflow and an additional `joblib.dump()` step to save the trained model and TF-IDF vectorizer.

Both notebooks use a 20,000-row sample by default for faster local execution. To use the entire cleaned dataset, set `SAMPLE_SIZE = None`.

## Classification problem

- Type: **Binary Classification / NLP**
- Labels: `0 = Fake`, `1 = Real`
- Features: headline + first 1,000 characters of article text
- Models: Multinomial Naive Bayes and Logistic Regression
- Metrics: Accuracy, weighted F1, classification report, confusion matrix
- Best model: selected automatically by weighted F1

## Data source

- [WELFake on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- [Original Zenodo record](https://zenodo.org/records/4561253)
- Verma et al. (2021), *WELFake: Word Embedding Over Linguistic Features for Fake News Detection*, [DOI](https://doi.org/10.1109/TCSS.2021.3068519)
- Related beginner project by Aman Kharwal: [Fake News Detection with Machine Learning](https://amanxai.com/2021/06/30/fake-news-detection-with-machine-learning/)

## Hugging Face Spaces

After running the Streamlit notebook, copy both `.pkl` files from `models/` into `huggingface_space/models/`. Then upload the content of `huggingface_space/` to a **Docker Space**. See `huggingface_space/README.md` for its configuration.

> **Important:** The example is educational. A text-classification prediction is not proof of whether a claim is true; the model does not check facts. Only load trusted `.pkl` files, because `joblib.load()` can execute code from untrusted files.
