from pathlib import Path
import joblib
import streamlit as st

st.set_page_config(page_title="Fake News Classification", page_icon="📰", layout="centered")

# Simple newspaper-inspired colors and typography.
st.markdown("""
<style>
.stApp { background-color: #F7F1E4; color: #202020; }
.block-container { max-width: 780px; padding-top: 2rem; }
h1, h2, h3 { font-family: Georgia, serif !important; color: #202020 !important; }
.newspaper-label { letter-spacing: 3px; font-size: 12px; font-weight: bold; color: #755D35; }
.newspaper-title { font-family: Georgia, serif; font-size: clamp(36px, 7vw, 56px); font-weight: bold; line-height: 1.1; }
.newspaper-rule { height: 3px; background: #222; margin: 15px 0 8px; }
.newspaper-subtitle { color: #554C42; font-size: 15px; }
.stButton > button[kind="primary"] { background: #D8AF4A; color: #171717; border: 1px solid #9B772C; font-weight: bold; }
.stButton > button[kind="primary"]:hover { background: #E6C46A; color: #171717; border-color: #9B772C; }
.stTextInput input, .stTextArea textarea { background: #FFFCF6; color: #202020; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_dir = Path(__file__).resolve().parent / "models"
    model = joblib.load(model_dir / "fake_news_model.pkl")
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
    return model, vectorizer

st.markdown('<div class="newspaper-label">THE NEWS DESK · NLP PROJECT</div>', unsafe_allow_html=True)
st.markdown('<div class="newspaper-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="newspaper-title">Fake News<br>Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="newspaper-rule"></div>', unsafe_allow_html=True)
st.markdown('<p class="newspaper-subtitle">Enter a headline and an article to see the model’s prediction.</p>', unsafe_allow_html=True)

headline = st.text_input("News headline", placeholder="Enter a headline...")
article = st.text_area("News article", placeholder="Paste the article text here...", height=200)

if st.button("Classify Article", type="primary", use_container_width=True):
    content = f"{headline.strip()} {article[:1000].strip()}".strip()
    if not content:
        st.warning("Please enter a headline or an article.")
    else:
        try:
            model, vectorizer = load_model()
            prediction = model.predict(vectorizer.transform([content]))[0]
            if prediction == 0:
                st.error("Model prediction: FAKE NEWS")
            else:
                st.success("Model prediction: REAL NEWS")
        except FileNotFoundError:
            st.error("Model files missing: add both .pkl files to the models folder.")
        except Exception as exc:
            st.error(f"Could not load the model: {exc}")

st.divider()
st.caption("Educational demo · Logistic Regression + TF-IDF · Predictions are not verified facts.")
