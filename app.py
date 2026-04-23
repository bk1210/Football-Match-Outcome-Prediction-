import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import os
from transformers import RobertaTokenizerFast, RobertaModel

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sarcasm Detector",
    page_icon="🧠",
    layout="centered"
)

# ── Config ───────────────────────────────────────────────────────────────────
MAX_LEN        = 128
MODEL_NAME     = "roberta-base"
SARCASM_THRESH = 0.60
DEVICE         = torch.device("cpu")   # HuggingFace free tier = CPU
MODEL_CKPT     = "best_model.pt"

SARCASM_ID2LABEL   = {0: "Not Sarcastic", 1: "Sarcastic 😏"}
SENTIMENT_ID2LABEL = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}

# ── Model Definition (must match training code exactly) ──────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class SarcasmDetector(nn.Module):
    def __init__(self, model_name, num_sarc=2, num_sent=3, dropout=0.15):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        H = self.roberta.config.hidden_size  # 768

        self.sarcasm_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(H, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_sarc))

        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(H, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_sent))

        self.confidence_gate = nn.Linear(H, 1)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return (self.sarcasm_head(cls), self.sentiment_head(cls),
                torch.sigmoid(self.confidence_gate(cls)))


# ── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s'!?.,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Load Model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    model = SarcasmDetector(MODEL_NAME)
    if os.path.exists(MODEL_CKPT):
        state = torch.load(MODEL_CKPT, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return tokenizer, model, True
    return tokenizer, model, False


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(text, tokenizer, model):
    cleaned = clean_text(text)
    enc = tokenizer(
        cleaned, max_length=MAX_LEN,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    sar_l, sen_l, conf = model(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE)
    )
    sar_p  = F.softmax(sar_l, dim=1)[0].cpu().numpy()
    sen_p  = F.softmax(sen_l, dim=1)[0].cpu().numpy()
    conf_v = conf[0].item()

    sar_id = 1 if sar_p[1] >= SARCASM_THRESH else 0
    sen_id = int(np.argmax(sen_p))

    # Sarcasm-Aware Sentiment Gate
    if sar_id == 1 and sar_p[1] > 0.65:
        final_sentiment = "Implicit Negative ⚠️"
    else:
        final_sentiment = SENTIMENT_ID2LABEL[sen_id]

    return {
        "sarcasm_label":     SARCASM_ID2LABEL[sar_id],
        "sarcasm_prob":      float(sar_p[1]),
        "not_sarcasm_prob":  float(sar_p[0]),
        "sentiment":         final_sentiment,
        "confidence":        conf_v,
        "is_sarcastic":      sar_id == 1,
    }


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 Sarcasm & Sentiment Detector")
st.markdown(
    "Multi-Task **RoBERTa** model detecting sarcasm and sentiment simultaneously. "
    "Features a **Sarcasm-Aware Sentiment Gate** that resolves implicit negation."
)
st.divider()

tokenizer, model, model_loaded = load_model()

if not model_loaded:
    st.error(
        "⚠️ `best_model.pt` not found! "
        "Please upload your trained model file to the same directory as `app.py`."
    )

# Example texts
st.subheader("Try an example")
examples = [
    "Study Finds That Watching TV All Day Is Actually Good For Your Health",
    "Man Who Texts During Movies Considerate For Sharing Screen Glow With Others",
    "Area Company Celebrates Record Profits By Cutting Employee Benefits Again",
    "City Council Approves Funding For New Public Library And Community Center",
    "Dog Reunited With Owner After Being Missing For Three Days",
]

cols = st.columns(2)
selected_example = None
for i, ex in enumerate(examples):
    col = cols[i % 2]
    if col.button(f"📝 {ex[:55]}…" if len(ex) > 55 else f"📝 {ex}", key=f"ex_{i}"):
        selected_example = ex

# Text input
st.subheader("Or type your own")
user_input = st.text_area(
    "Enter text to analyse:",
    value=selected_example if selected_example else "",
    height=100,
    placeholder="e.g. Oh great, another Monday. Just what I needed."
)

if st.button("🔍 Analyse", type="primary", disabled=not model_loaded):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analysing..."):
            result = predict(user_input, tokenizer, model)

        st.divider()
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            color = "🔴" if result["is_sarcastic"] else "🟢"
            st.metric(
                label="Sarcasm",
                value=result["sarcasm_label"],
                delta=f"{color} {result['sarcasm_prob']*100:.1f}% confidence"
            )

        with col2:
            st.metric(
                label="Sentiment",
                value=result["sentiment"]
            )

        with col3:
            st.metric(
                label="Model Confidence",
                value=f"{result['confidence']*100:.1f}%"
            )

        # Probability bar
        st.divider()
        st.subheader("Sarcasm Probability Breakdown")
        col_a, col_b = st.columns(2)
        col_a.progress(result["sarcasm_prob"], text=f"Sarcastic: {result['sarcasm_prob']*100:.1f}%")
        col_b.progress(result["not_sarcasm_prob"], text=f"Not Sarcastic: {result['not_sarcasm_prob']*100:.1f}%")

        if result["is_sarcastic"]:
            st.info("💡 **Sarcasm-Aware Sentiment Gate** activated — sentiment may differ from literal meaning.")

st.divider()
st.markdown(
    "Built by **Bharath Kesav R** · "
    "[GitHub](https://github.com/bk1210) · "
    "Model: Multi-Task RoBERTa (F1-Macro 0.977, AUC 0.997)"
)
