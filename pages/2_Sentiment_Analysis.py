import streamlit as st
import joblib
import re
import os

MODEL_PATH = os.path.join( 'notebooks', 'model_assets', 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join( 'notebooks', 'model_assets', 'tfidf_vectorizer.pkl')
model = None
tfidf_vectorizer = None

@st.cache_resource
def load_assets():
    """Load the trained model and vectorizer with Streamlit caching."""
    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_vectorizer = joblib.load(VECTORIZER_PATH)
        return loaded_model, loaded_vectorizer
    except FileNotFoundError:
        st.error(f"Error: Model or Vectorizer file not found at the expected location.")
        st.info(f"Attempted to load model from: {MODEL_PATH}")

        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during asset loading: {e}")
        return None, None


model, tfidf_vectorizer = load_assets()



def clean_tweet(text):
    text = str(text) 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'[^\w\s#]', '', text)
    text = text.lower()
    return text



def predict_sentiment(text):

    if model is None or tfidf_vectorizer is None:
        return "Assets Not Loaded"
        
    cleaned_text = clean_tweet(text)
    
    if not cleaned_text.strip():
        return "Neutral (Empty Input)"
        
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    
    prediction = model.predict(text_tfidf)[0]
    
    sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    
    return sentiment_map.get(prediction, "Neutral") 

# --- 4. Streamlit UI Design ---

st.set_page_config(page_title="Twitter Sentiment Classifier", layout="centered")

st.title("🐦 Sentiment Analysis Tool")
st.markdown("---")

# Input box for the user
user_input = st.text_area(
    "Enter a tweet or a short political comment:",
    "Modi's new policy proposal looks promising, great job by the government!",
    height=150
)

# Button to trigger the prediction
if st.button("Analyze Sentiment", help="Click to classify the text using the trained model"):
    
    # Check if assets failed to load
    if model is None or tfidf_vectorizer is None:
        st.error("Cannot perform analysis. Model assets failed to load. Check console for path errors.")
    elif user_input:
        result = predict_sentiment(user_input)
        
        st.subheader("Analysis Result:")
        
        if result == "Positive":
            st.success(f"**Predicted Sentiment:** {result} 👍")
        elif result == "Negative":
            st.error(f"**Predicted Sentiment:** {result} 👎")
        elif result.startswith("Neutral"):
            st.info(f"**Predicted Sentiment:** {result} ↔️")
        else:
             st.warning(f"**Prediction Failed:** {result}")
            
        st.markdown("---")
        st.markdown(f"**Cleaned Text Used for Prediction:** `{clean_tweet(user_input)}`")
        
    else:
        st.warning("Please enter some text to analyze.")