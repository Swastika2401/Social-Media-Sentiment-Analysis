import streamlit as st

st.set_page_config(
    page_title="Twitter Sentiment Classifier Home",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Welcome to the 🐦 Twitter Sentiment Classifier")

st.markdown("""
---
### Project Overview

This is a demonstration of a **Natural Language Processing (NLP)** minor project.

We used real-world social media data related to prominent Indian political figures (**Narendra Modi**, **Rahul Gandhi**, and **Arvind Kejriwal**) to build a model that can automatically determine the sentiment (Positive, Neutral, or Negative) of a given piece of text.

### Key Technologies Used

* **Data Preparation:** Pandas, Regular Expressions (`re`)
* **Feature Engineering:** **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency)
* **Machine Learning Model:** **Logistic Regression** (a robust and efficient classification algorithm)
* **Deployment:** **Streamlit** (for the interactive web UI)

### How to Use

1.  Navigate to the **'Sentiment Analysis'** link in the sidebar on the left.
2.  Paste any tweet or political commentary into the text box.
3.  Click **'Analyze Sentiment'** to see the model's prediction.

### About the Model
The model was trained on a highly imbalanced dataset and achieved an overall accuracy of over 94% on the test set.

---
""")