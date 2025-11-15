import streamlit as st
import joblib
import pandas as pd
import re
import os

# --- 1. Load Model Assets ---
# Define paths relative to the app.py file
MODEL_PATH = os.path.join('notebooks', 'model_assets', 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join('notebooks', 'model_assets', 'tfidf_vectorizer.pkl')

@st.cache_resource
def load_assets():
    """Load the trained model and vectorizer with Streamlit caching."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Error: Model or Vectorizer file not found in the 'model_assets' directory.")
        st.stop()

model, tfidf_vectorizer = load_assets()

# --- 2. Preprocessing Function (from your original script) ---

def clean_tweet(text):
    """Applies the same cleaning steps used during training."""
    text= str(text) 
    text=re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'RT[\s]+', '', text) # Remove Retweet tags
    text = re.sub(r'[^\w\s#]', '', text) # Remove special characters/emojis
    text = text.lower() # Convert to lowercase
    return text

# --- 3. Prediction Function ---

def predict_sentiment(text):
    """Cleans, vectorizes, and predicts the sentiment of the input text."""
    
    # 1. Clean the text using the exact function used in training
    cleaned_text = clean_tweet(text)
    
    # 2. Vectorize the cleaned text using the loaded TF-IDF vectorizer
    # We pass [cleaned_text] because the vectorizer expects an iterable (list)
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    
    # 3. Predict the sentiment label (0, 1, or -1)
    prediction = model.predict(text_tfidf)[0]
    
    # 4. Map the numeric label to a human-readable string
    sentiment_map = {
        1: "Positive",
        0: "Neutral",
        -1: "Negative"
    }
    
    return sentiment_map.get(prediction, "Neutral") # Default to Neutral if prediction is unexpected

# --- 4. Streamlit UI Design ---

st.set_page_config(page_title="Twitter Sentiment Classifier", layout="centered")

st.title("🐦 Twitter Sentiment Classifier")
st.markdown("""
    This application uses a **Logistic Regression** model trained on real-world tweets to classify input text 
    as **Positive**, **Neutral**, or **Negative**.
""")

# Input box for the user
user_input = st.text_area(
    "Enter a tweet or a short political comment:",
    "Modi's new policy proposal looks promising, great job by the government!",
    height=150
)

# Button to trigger the prediction
if st.button("Analyze Sentiment", help="Click to classify the text using the trained model"):
    
    if user_input:
        # Get the prediction
        result = predict_sentiment(user_input)
        
        # Display the result with appropriate formatting
        st.subheader("Analysis Result:")
        
        if result == "Positive":
            st.success(f"**Predicted Sentiment:** {result} 👍")
        elif result == "Negative":
            st.error(f"**Predicted Sentiment:** {result} 👎")
        else:
            st.info(f"**Predicted Sentiment:** {result} ↔️")
            
        st.markdown("---")
        st.markdown(f"**Cleaned Text Used for Prediction:** `{clean_tweet(user_input)}`")
        
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("""
---
*Minor Project Demonstration using TF-IDF and Logistic Regression.*
""")