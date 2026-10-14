# Fake News Classification — Streamlit Demo

This folder contains a simple Streamlit app and the completed training notebook.

## Run in VS Code

1. Open this folder in VS Code.
2. Copy your two trained files from your existing project's `models/` folder into this project's `models/` folder:
   - `fake_news_model.pkl`
   - `tfidf_vectorizer.pkl`
3. In the VS Code terminal run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Hugging Face Spaces

1. Copy the same two `.pkl` files into `huggingface_space/models/`.
2. Create a new Space at https://huggingface.co/new-space with **Docker** SDK and **Public** visibility if desired.
3. Upload the **contents** of `huggingface_space/` to the root of your Space (not the outer folder).
4. Wait for build status **Running**, then open the Space app.

The uploaded notebook was already run by the user. This package does not contain the saved model binaries; they reside on the user's computer. Do not upload the CSV dataset to the Space.

## Dataset and method

WELFake Dataset: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Training combines title with the first 1,000 characters of article text and uses TF-IDF. The winning trained model is saved with joblib. The app repeats the same input construction and loads the saved model.

**Disclaimer:** Predictions are learned text patterns; they do not constitute fact-checking.
