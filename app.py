import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import string
import os

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import textstat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pdfplumber
import docx

nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('stopwords',   quiet=True)
nltk.download('wordnet',     quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_LEN   = 300
MAX_VOCAB = 20000

MODEL_DIR = "models"

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(f"{MODEL_DIR}/svm_model.pkl",            "rb") as f: svm       = pickle.load(f)
    with open(f"{MODEL_DIR}/decision_tree_model.pkl",  "rb") as f: dt        = pickle.load(f)
    with open(f"{MODEL_DIR}/adaboost_model.pkl",       "rb") as f: ada       = pickle.load(f)
    with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl",     "rb") as f: tfidf     = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl",               "rb") as f: scaler    = pickle.load(f)
    with open(f"{MODEL_DIR}/tokenizer.pkl",            "rb") as f: tokenizer = pickle.load(f)

    fnn  = keras.models.load_model(f"{MODEL_DIR}/fnn_model.h5")
    lstm = keras.models.load_model(f"{MODEL_DIR}/lstm_model.h5")
    cnn  = keras.models.load_model(f"{MODEL_DIR}/cnn_model.h5")

    return svm, dt, ada, tfidf, scaler, tokenizer, fnn, lstm, cnn

svm_model, dt_model, ada_model, tfidf, scaler, tokenizer, fnn_model, lstm_model, cnn_model = load_models()

# ── Preprocessing ─────────────────────────────────────────────────────────────
lemmatizer     = WordNetLemmatizer()
stop_words_set = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>',          '', text)
    text = re.sub(r'\[.*?\]',        '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+',            '', text)
    text = re.sub(r'\s+',           ' ', text).strip()
    return text

def extract_linguistic_features(text):
    raw   = str(text)
    words = raw.split()
    sents = sent_tokenize(raw)
    n_words = max(len(words), 1)
    n_sents = max(len(sents), 1)
    n_chars = max(len(raw),   1)

    return np.array([[
        np.mean([len(w) for w in words]) if words else 0,
        len(set(w.lower() for w in words)) / n_words,
        n_words / n_sents,
        sum(1 for c in raw if c in string.punctuation) / n_chars,
        sum(1 for w in words if w.lower() in stop_words_set) / n_words,
        textstat.flesch_reading_ease(raw),
        textstat.flesch_kincaid_grade(raw),
        sum(1 for c in raw if c.isupper()) / n_chars,
        len(raw),
        n_words,
        n_sents,
    ]])

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join(para.text for para in doc.paragraphs)

def predict_all(text):
    cleaned   = clean_text(text)
    tfidf_vec = tfidf.transform([cleaned])
    ling_feat = scaler.transform(extract_linguistic_features(text))

    import scipy.sparse as sp
    combined       = sp.hstack([tfidf_vec, sp.csr_matrix(ling_feat)])
    combined_dense = combined.toarray().astype(np.float32)  # cast to float32

    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    results = {}

    # FNN separately with float32 input
    fnn_prob = float(fnn_model(combined_dense, training=False).numpy()[0][0])
    results['FNN'] = {'prediction': int(fnn_prob > 0.5), 'confidence': fnn_prob}

    for name, model, X in [
        ('SVM',           svm_model, tfidf_vec),
        ('Decision Tree', dt_model,  tfidf_vec),
        ('AdaBoost',      ada_model, tfidf_vec),
    ]:
        prob = model.predict_proba(X)[0][1]
        results[name] = {'prediction': int(prob > 0.5), 'confidence': prob}

    for name, model, X in [
        ('LSTM', lstm_model, padded),
        ('CNN',  cnn_model,  padded),
    ]:
        prob = float(model(X, training=False).numpy()[0][0])
        results[name] = {'prediction': int(prob > 0.5), 'confidence': prob}

    return results

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 AI vs Human")
    st.markdown("---")
    st.markdown("### Model Selector")
    selected_model = st.selectbox(
        "Choose a model for primary prediction",
        ["FNN", "SVM", "AdaBoost", "LSTM", "CNN", "Decision Tree"]
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This app detects whether text was written by a human or generated by AI. "
        "It uses six trained classifiers and compares their predictions."
    )
    st.markdown("---")
    st.markdown("**CS-5331 | Texas Tech University**")
    st.markdown("Summer I 2026")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 AI vs Human Text Detector")
st.markdown("Upload a PDF or Word document, or paste text below. The app will predict whether it was written by a human or generated by AI.")
st.markdown("---")

# ── Input section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload a Document")
    uploaded_file = st.file_uploader("Upload PDF or Word file", type=["pdf", "docx"])

with col2:
    st.markdown("### Or Paste Text")
    input_text = st.text_area("Paste your text here", height=200, placeholder="Enter text to analyze...")

# Resolve input
text_input = ""
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        text_input = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        text_input = extract_text_from_docx(uploaded_file)
    st.success(f"File loaded: {uploaded_file.name} ({len(text_input.split())} words)")
elif input_text.strip():
    text_input = input_text.strip()

# ── Analyze button ────────────────────────────────────────────────────────────
analyze = st.button("Analyze Text", type="primary", use_container_width=True)

if analyze and text_input:
    with st.spinner("Running predictions..."):
        results = predict_all(text_input)

    st.markdown("---")

    # ── Primary prediction ────────────────────────────────────────────────────
    primary = results[selected_model]
    label   = "AI Generated" if primary['prediction'] == 1 else "Human Written"
    conf    = primary['confidence'] if primary['prediction'] == 1 else 1 - primary['confidence']
    color   = "#DD8452" if primary['prediction'] == 1 else "#4C72B0"

    st.markdown("## Prediction Result")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(
            f"<div style='text-align:center; padding:20px; border-radius:10px; background-color:{color}22; border: 2px solid {color}'>"
            f"<h2 style='color:{color}'>{label}</h2>"
            f"<p style='font-size:18px'>Confidence: <b>{conf:.1%}</b></p>"
            f"<p style='color:gray'>Model: {selected_model}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(conf * 100, 1),
            title={'text': "Confidence Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': color},
                'steps': [
                    {'range': [0,  50], 'color': '#f0f0f0'},
                    {'range': [50, 75], 'color': '#ffe0cc'},
                    {'range': [75, 100],'color': '#ffcba4'},
                ],
            }
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        words = len(text_input.split())
        sents = len(sent_tokenize(text_input))
        st.markdown("#### Text Statistics")
        st.metric("Word Count",     words)
        st.metric("Sentence Count", sents)
        st.metric("Avg Words/Sentence", round(words / max(sents, 1), 1))
        st.metric("Readability (Flesch)", round(textstat.flesch_reading_ease(text_input), 1))

    st.markdown("---")

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown("## All Model Predictions")

    comp_data = []
    for name, res in results.items():
        c = res['confidence'] if res['prediction'] == 1 else 1 - res['confidence']
        comp_data.append({
            'Model'      : name,
            'Prediction' : 'AI' if res['prediction'] == 1 else 'Human',
            'Confidence' : f"{c:.1%}",
            'AI Score'   : round(res['confidence'], 4),
        })

    comp_df = pd.DataFrame(comp_data)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with col2:
        fig2 = px.bar(
            comp_df, x='Model', y='AI Score',
            color='Prediction',
            color_discrete_map={'AI': '#DD8452', 'Human': '#4C72B0'},
            title='AI Probability Score by Model',
            range_y=[0, 1]
        )
        fig2.add_hline(y=0.5, line_dash='dash', line_color='gray')
        fig2.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Linguistic features ───────────────────────────────────────────────────
    st.markdown("## Text Analysis")

    ling = extract_linguistic_features(text_input)[0]
    feat_names = [
        'Avg Word Length', 'Type Token Ratio', 'Avg Sentence Length',
        'Punctuation Density', 'Stopword Ratio', 'Flesch Reading Ease',
        'Flesch Kincaid Grade', 'Uppercase Ratio', 'Char Count',
        'Word Count', 'Sentence Count'
    ]

    ling_df = pd.DataFrame({'Feature': feat_names, 'Value': ling})

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(ling_df, use_container_width=True, hide_index=True)

    with col2:
        fig3 = px.bar(
            ling_df.head(8), x='Value', y='Feature',
            orientation='h',
            title='Linguistic Feature Values',
            color='Value',
            color_continuous_scale='Blues'
        )
        fig3.update_layout(height=350, margin=dict(t=40, b=0), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── TF-IDF top terms ──────────────────────────────────────────────────────
    st.markdown("## Feature Importance")
    st.markdown("Top TF-IDF terms from your input text that influenced the prediction.")

    cleaned   = clean_text(text_input)
    tfidf_vec = tfidf.transform([cleaned])
    feat_arr  = np.asarray(tfidf_vec.todense()).flatten()
    top_idx   = feat_arr.argsort()[::-1][:15]
    top_terms = np.array(tfidf.get_feature_names_out())[top_idx]
    top_vals  = feat_arr[top_idx]

    fig4 = px.bar(
        x=top_vals, y=top_terms,
        orientation='h',
        title='Top 15 TF-IDF Terms in Input Text',
        labels={'x': 'TF-IDF Score', 'y': 'Term'},
        color=top_vals,
        color_continuous_scale='Blues'
    )
    fig4.update_layout(height=400, margin=dict(t=40, b=0), showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ── Download report ───────────────────────────────────────────────────────
    st.markdown("## Download Report")

    report_lines = [
        "AI vs Human Text Detection Report",
        "=" * 40,
        f"Primary Model    : {selected_model}",
        f"Prediction       : {label}",
        f"Confidence       : {conf:.1%}",
        "",
        "All Model Predictions:",
        "-" * 40,
    ]
    for name, res in results.items():
        c = res['confidence'] if res['prediction'] == 1 else 1 - res['confidence']
        p = 'AI' if res['prediction'] == 1 else 'Human'
        report_lines.append(f"{name:<15}: {p}  ({c:.1%})")

    report_lines += [
        "",
        "Text Statistics:",
        "-" * 40,
        f"Word Count       : {len(text_input.split())}",
        f"Sentence Count   : {len(sent_tokenize(text_input))}",
        f"Flesch Ease      : {round(textstat.flesch_reading_ease(text_input), 2)}",
        f"Flesch Grade     : {round(textstat.flesch_kincaid_grade(text_input), 2)}",
        "",
        "Top TF-IDF Terms:",
        "-" * 40,
        ", ".join(top_terms),
    ]

    report = "\n".join(report_lines)
    st.download_button(
        label="Download Report as .txt",
        data=report,
        file_name="detection_report.txt",
        mime="text/plain",
        use_container_width=True
    )

elif analyze and not text_input:
    st.warning("Please upload a file or paste some text before clicking Analyze.")