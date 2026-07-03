import streamlit as st
import joblib
import pandas as pd
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


TEXT_COLUMNS = (
    "text",
    "clean_text",
    "tweet",
    "full_text",
    "tweetText",
    "tweet_text",
    "reply_text",
    "replyText",
    "content",
    "body",
)


def find_text_column(dataframe):
    normalized = {str(column).strip().lower(): column for column in dataframe.columns}
    for column in TEXT_COLUMNS:
        found = normalized.get(column.lower())
        if found is not None:
            return found
    return None


def classify_uploaded_tweets(dataframe):
    text_column = find_text_column(dataframe)
    if text_column is None:
        raise ValueError(
            "CSV must include text, clean_text, tweet, full_text, tweetText, "
            "reply_text, replyText, content, or body."
        )

    results = dataframe.copy()
    results["xquik_text"] = results[text_column].fillna("").astype(str).str.strip()
    results = results[results["xquik_text"] != ""]
    if results.empty:
        raise ValueError("The selected text column is empty.")

    results["clean_text"] = results["xquik_text"].apply(clean_tweet)
    results["predicted_sentiment"] = results["xquik_text"].apply(predict_sentiment)
    return results

# --- 4. Streamlit UI Design ---

st.set_page_config(page_title="Twitter Sentiment Classifier", layout="centered")

st.title("🐦 Sentiment Analysis Tool")
st.markdown("---")

st.subheader("Batch Analyze Xquik/TweetClaw Export")
uploaded_file = st.file_uploader(
    "Upload a reviewed tweet CSV",
    type=("csv",),
    help="Accepts common text fields such as tweetText, reply_text, full_text, text, and clean_text.",
)

if uploaded_file is not None:
    if model is None or tfidf_vectorizer is None:
        st.error("Cannot perform batch analysis. Model assets failed to load.")
    else:
        try:
            batch_results = classify_uploaded_tweets(pd.read_csv(uploaded_file))
            st.success(f"Analyzed {len(batch_results)} rows.")
            st.dataframe(batch_results[["xquik_text", "clean_text", "predicted_sentiment"]])
            st.download_button(
                "Download Predictions CSV",
                data=batch_results.to_csv(index=False),
                file_name="xquik_tweet_sentiment_predictions.csv",
                mime="text/csv",
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
            st.error(str(exc))

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
