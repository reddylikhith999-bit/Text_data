"""
Predict whether an Amazon review is positive or negative.

This app uses the trained machine learning model saved by train.py.
Run it with: streamlit run app.py
"""

from pathlib import Path

import joblib
import streamlit as st
from sklearn.feature_extraction.text import HashingVectorizer


# Find the trained model in the same folder as this app.
MODEL_PATH = Path(__file__).parent / "sentiment_model.joblib"


@st.cache_resource
def load_model(model_path):
    """Load the trained classifier and its text feature converter."""
    saved_data = joblib.load(model_path)

    # Convert review words into numeric features using the same settings as train.py.
    vectorizer = HashingVectorizer(
        n_features=saved_data["n_features"],
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
    )

    return saved_data["model"], vectorizer


st.set_page_config(
    page_title="🛍️ Review Sentiment",
    page_icon="🛍️",
)

st.title("🛍️ Amazon Review Sentiment")
st.write("Enter a review to predict whether its sentiment is positive or negative.")

if not MODEL_PATH.is_file():
    st.error("Trained model not found. Run train.py first.")
    st.stop()

try:
    model, vectorizer = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

review = st.text_area(
    "Review text",
    placeholder="Type or paste a product review here...",
    height=160,
)

if st.button("Predict sentiment", type="primary"):
    if not review.strip():
        st.warning("Enter a review first.")
    else:
        # Convert the review text to numeric features and predict its sentiment.
        review_features = vectorizer.transform([review.strip()])
        prediction = int(model.predict(review_features)[0])

        if prediction == 1:
            st.success("Positive review")
        else:
            st.error("Negative review")

        # Display the probability for each sentiment when available.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(review_features)[0]
            st.write(f"Negative: {probabilities[0]:.1%}")
            st.write(f"Positive: {probabilities[1]:.1%}")