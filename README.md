 Social Media Sentiment Classifier (Political Tweets)

## Project Overview

This project implements a Natural Language Processing (NLP) pipeline to classify the sentiment of political tweets related to prominent Indian political figures (Narendra Modi, Rahul Gandhi, and Arvind Kejriwal). The goal is to demonstrate text preprocessing, feature engineering, and classification using classical Machine Learning techniques, culminating in a simple, interactive web deployment.

### 🎯 Key Objectives

1. Load and combine raw social media data
2. Clean and preprocess raw tweet text
3. Apply a dictionary-based method for initial sentiment labeling
4. Train an efficient classification model (Logistic Regression)
5. Deploy the model via a multi-page web application using Streamlit

---

## 🛠 Technology Stack

| Category | Tool / Library | Purpose |
|:---------|:--------------|:--------|
| *Language* | Python (3.11+) | Core programming language |
| *Data Handling* | Pandas, NumPy | Data manipulation and storage |
| *Preprocessing* | re (Regular Expressions) | Cleaning URLs, mentions, and special characters |
| *Feature Engineering* | *TfidfVectorizer* (Scikit-learn) | Converting text data into numerical features |
| *Model* | *Logistic Regression* (Scikit-learn) | Multi-class classification model for sentiment prediction |
| *Deployment* | *Streamlit* | Creating the interactive, user-friendly web interface |
| *Versioning* | Git / GitHub | Code management and collaboration |

---

## 🚀 Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.11+
- pip (Python package installer)

### 1. Clone the Repository

bash
git clone https://github.com/Swastika2401/Social-Media-Sentiment-Analysis.git
cd Social-Media-Sentiment-Analysis


### 2. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment:

bash
# Create the environment
python3 -m venv venv311

# Activate the environment (Linux/macOS)
source venv311/bin/activate

# Activate the environment (Windows)
venv311\Scripts\activate


### 3. Install Dependencies

Install all necessary Python packages:

bash
pip install pandas scikit-learn joblib streamlit


### 4. Data and Model Assets

The project assumes the following directory structure for model files:

| File/Folder | Status |
|:------------|:-------|
| data/raw/ | Requires raw CSV files (Modi, Gandhi, Kejriwal data) |
| notebooks/model_assets/ | *Contains pre-trained model assets* |

*Note: The model (sentiment_model.pkl) and vectorizer (tfidf_vectorizer.pkl) are saved in the notebooks/model_assets/ folder and are ready for deployment.*

---

## ⚙ Running the Application

The application is run via Streamlit, which automatically handles the multi-page structure.

1. Ensure your virtual environment is active
2. Run the main application file (1_Home.py):

bash
streamlit run 1_Home.py


3. The application will open in your browser (usually at http://localhost:8501)
4. Use the *sidebar* to navigate to the *"Sentiment Analysis"* tool page

---

## 💡 Methodology

### 1. Data Labeling

Raw tweets were initially labeled using a simple *dictionary-based approach, mapping custom lists of Hindi/English **positive* and *negative* keywords to assign labels:
- 1: Positive
- 0: Neutral
- -1: Negative

### 2. Preprocessing

The text underwent crucial cleaning steps:
- Removal of URLs, Retweet tags (RT), and user mentions (@user)
- Removal of special characters and emojis
- Conversion to lowercase

### 3. Feature Engineering

The cleaned text was converted into a feature matrix using the *TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer*. This technique weighs the importance of a word based on its frequency in a specific tweet relative to its frequency across the entire dataset.

### 4. Modeling

A *Logistic Regression* classifier was chosen for its interpretability and efficiency in high-dimensional sparse data (like TF-IDF vectors).

---

## ✅ Evaluation Results

The model was evaluated on a held-out test set (20% of the total data).

| Metric | Score | Note |
|:-------|:------|:-----|
| *Overall Accuracy* | *94.07%* | High, but influenced by the large number of Neutral tweets |
| *F1-Score (Positive)* | *0.92* | Strong balance between precision and recall for positive sentiment |
| *F1-Score (Negative)* | *0.83* | The weakest class, indicating some actual negative tweets were missed (classified as Neutral) |
| *F1-Score (Neutral)* | *0.96* | Excellent performance on the dominant class |

---

## 📁 Project Structure


Social-Media-Sentiment-Analysis/
├── 1_Home.py                      # Main Streamlit application entry point
├── pages/                         # Additional Streamlit pages
│   └── 2_Sentiment_Analysis.py   # Sentiment analysis tool page
├── data/
│   └── raw/                      # Raw CSV files for tweets
|   └── processed/                # combined  CSV files for tweets
├── notebooks/
│   └── model_assets/             # Pre-trained model and vectorizer
│       ├── sentiment_model.pkl
│       └── tfidf_vectorizer.pkl
├── venv311/                      # Virtual environment (not tracked in git)
├── requirements.txt              # Python dependencies
└── README.md                     # This file


---

## 🧑‍💻 Project Status and Future Work

### Status
✅ *Complete* (Data Cleaning, Modeling, Deployment)

## 🤝 Contributing

This is a college minor project. If you'd like to suggest improvements:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/improvement)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature/improvement)
5. Open a Pull Request

---

## 📝 License

This project is created for educational purposes as part of a college minor project.

---

## 👨‍💻 Contact

*Developer1:* [Anurag Kumar / 231B058@juetguna.in]  
*Developer2:* [Roshani Singh / 231B271@juetguna.in]  
*Developer3:* [Swastika Kumari / 231B355@juetguna.in]  
*Institution:* [Jaypee University of Engineering and Technology]  
*Year:* [3rd]

---

## 🙏 Acknowledgments

- Dataset sources for political tweet analysis
- Scikit-learn documentation and community
- Streamlit for the excellent deployment framework
- Course instructors and mentors

---

