from pathlib import Path
from urllib.parse import urlparse
import ipaddress
import socket

import joblib
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="Fake News Classification", page_icon="📰", layout="centered")

# Newspaper-inspired design
st.markdown("""
<style>
.stApp { background-color: #F7F1E4; color: #202020; }
.block-container { max-width: 780px; padding-top: 5rem; }
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


def read_news_url(url):
    """Best-effort HTML extraction for publicly accessible news pages."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a valid http:// or https:// URL.")
    # Public hosting: do not allow fetches to local/private/internal addresses.
    hostname = parsed.hostname
    addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Only public news websites are supported.")
    response = requests.get(url, timeout=10, allow_redirects=False,
                            headers={"User-Agent": "Mozilla/5.0 (educational news classifier)"})
    if 300 <= response.status_code < 400:
        raise ValueError("This website redirects. Please use its final article URL.")
    response.raise_for_status()
    if "text/html" not in response.headers.get("Content-Type", "").lower():
        raise ValueError("The URL does not point to a regular HTML page.")
    if len(response.content) > 2_000_000:
        raise ValueError("This page is too large to analyze.")
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    article = soup.find("article")
    paragraphs = (article or soup).find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    if len(text.strip()) < 80:
        raise ValueError("Could not extract enough article text. Please paste the text manually.")
    return title, text


st.markdown('<div class="newspaper-label">THE NEWS DESK · NLP PROJECT</div>', unsafe_allow_html=True)
st.markdown('<div class="newspaper-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="newspaper-title">📰 Fake News<br><span style="margin-left: 90px;">Classification</span></div>', unsafe_allow_html=True)
st.markdown('<div class="newspaper-rule"></div>', unsafe_allow_html=True)
st.markdown('<p class="newspaper-subtitle">Choose how to provide the news and see the model’s prediction</p>', unsafe_allow_html=True)

method = st.radio("News input method", ["Enter News Manually", "Paste News URL", "Upload News File"], horizontal=True)

headline = ""
article = ""
url = ""
upload = None

if method == "Enter News Manually":
    headline = st.text_input("News headline", placeholder="Enter a headline...")
    article = st.text_area("News article", placeholder="Paste the article text here...", height=200)
elif method == "Paste News URL":
    url = st.text_input("News URL", placeholder="https://example.com/news/article")
    st.caption("Only publicly accessible English articles. Some websites block automatic extraction.")
else:
    upload = st.file_uploader("Upload a news text file", type=["txt"])
    st.caption("Upload a UTF-8 .txt file containing an English news article.")

if st.button("Classify News", type="primary", use_container_width=True):
    try:
        if method == "Paste News URL":
            if not url.strip():
                st.warning("Please enter a news URL.")
                st.stop()
            headline, article = read_news_url(url.strip())
        elif method == "Upload News File":
            if upload is None:
                st.warning("Please upload a .txt file.")
                st.stop()
            if upload.size > 1_000_000:
                raise ValueError("Please upload a text file smaller than 1 MB.")
            article = upload.getvalue().decode("utf-8-sig")
        content = f"{headline.strip()} {article[:1000].strip()}".strip()
        if not content:
            st.warning("Please enter or upload some news text.")
        else:
            model, vectorizer = load_model()
            prediction = model.predict(vectorizer.transform([content]))[0]
            if prediction == 0:
                st.error("Model prediction: FAKE NEWS")
            else:
                st.success("Model prediction: REAL NEWS")
    except FileNotFoundError:
        st.error("Model files are missing. Add both .pkl files to the models folder.")
    except (ValueError, UnicodeDecodeError, requests.RequestException, socket.gaierror) as exc:
        st.warning(f"Could not read this news item: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Educational demo · Logistic Regression + TF-IDF · Predictions are not verified facts.")