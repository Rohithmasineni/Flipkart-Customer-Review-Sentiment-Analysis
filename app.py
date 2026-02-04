import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page Configuration
st.set_page_config(
    page_title="Flipkart Customer Review Predictor",
    page_icon="🛒",
    layout="centered"
)

# Model Path
MODEL_PATH = r"C:\Users\rohit\Downloads\INNOMATICS\DS with Adv Gen AI Internship\MLOps Tasks\Flipkart Sentiment Analysis\model"
LABEL_MAP = {0: "Negative", 1: "Positive"}

# Load Model & Tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# App UI
st.markdown("## 🛒 Flipkart Customer Review Predictor")

st.markdown(
    "Write a Flipkart review below to get instant sentiment feedback."
)

st.markdown("### Enter Customer Review")
user_input = st.text_area(
    "",
    height=100,
    placeholder="Example: The product quality is amazing and delivery was quick!"
)

# Prediction Function
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    sentiment = np.argmax(probs)
    confidence = probs[sentiment]

    return sentiment, confidence

# Action Button
if st.button("Analyze Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review before clicking analyze.")
    else:
        with st.spinner("Analyzing customer sentiment..."):
            sentiment, confidence = predict_sentiment(user_input)

        st.success("Analysis Complete!")

        # Result Section
        st.markdown("#### Sentiment Result:")

        if sentiment == 1:
            st.markdown("### Positive Review 😊📈")
        else:
            st.markdown("### Negative Review 😞📉")

        # st.write(f"**Confidence:** {confidence:.2%}")

# Footer
st.markdown("---")
st.caption("Built for Flipkart Review Analysis • Powered by AI")


