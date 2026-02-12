# ==============================
# Customer Review Analytics & NLP Pipeline
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib 

# ------------------------------
# Load Saved Model & Vectorizer
# ------------------------------
model = joblib.load("models/rf_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Customer Review Analytics & NLP Pipeline",
    layout="wide",
    page_icon="📊"
)

# ------------------------------
# HEADER
# ------------------------------
st.title("📊 Customer Review Analytics & NLP Pipeline")
st.markdown("End-to-End NLP & Machine Learning system for customer review analysis and predictive modeling.")

st.markdown("---")

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
section = st.sidebar.radio(
    "Navigation",
    [
        "📌 Project Overview",
        "📈 EDA Insights",
        "😊 Sentiment Analysis",
        "🧠 Topic Modeling",
        "🤖 Model Performance",
        "🔮 Live Prediction"
    ]
)

# ------------------------------
# PROJECT OVERVIEW
# ------------------------------
if section == "📌 Project Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", "23,000+")
    col2.metric("Models Built", "4")
    col3.metric("Deployment", "Streamlit")

    st.markdown("---")

    st.subheader("Pipeline Components")

    st.markdown("""
    **Data Processing**
    - Text Cleaning & Normalization
    - Stopword Removal
    - Feature Engineering
    
    **Modeling**
    - Sentiment Analysis (VADER)
    - Topic Modeling (LDA)
    - Logistic Regression
    - Random Forest Classifier
    - Linear Regression (Rating Prediction)
    
    **Evaluation**
    - Precision / Recall / F1
    - Confusion Matrix
    - MAE / MSE
    """)

# ------------------------------
# EDA SECTION
# ------------------------------
elif section == "📈 EDA Insights":

    st.subheader("Exploratory Data Analysis")

    st.markdown("""
    Key insights explored:
    - Rating distribution patterns
    - Age vs Rating trends
    - Category & Subcategory performance
    - Channel & Location behavior
    """)

    st.info("📌 Add saved EDA charts here (matplotlib or seaborn exports).")

# ------------------------------
# SENTIMENT ANALYSIS
# ------------------------------
elif section == "😊 Sentiment Analysis":

    st.subheader("Sentiment Analysis Using VADER")

    col1, col2 = st.columns(2)

    col1.metric("Positive Reviews", "82%")
    col2.metric("Negative Reviews", "18%")

    st.markdown("""
    - Used VADER compound score
    - Compared sentiment polarity with rating-based classification
    - Identified mismatched sentiment patterns
    """)

    st.info("📌 Add sentiment distribution chart here.")

# ------------------------------
# TOPIC MODELING
# ------------------------------
elif section == "🧠 Topic Modeling":

    st.subheader("Latent Dirichlet Allocation (LDA)")

    st.markdown("""
    Discovered dominant customer discussion themes:
    
    - Fit & Size Issues  
    - Fabric & Material Quality  
    - Dress & Design Appeal  
    - Comfort & Wearability  
    - Color & Style Preferences  
    """)

    st.success("5 dominant topics extracted using CountVectorizer + LDA.")

    st.info("📌 Add topic distribution visualization here.")

# ------------------------------
# MODEL PERFORMANCE
# ------------------------------
elif section == "🤖 Model Performance":

    st.subheader("Classification & Regression Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Random Forest Accuracy", "94%")
    col2.metric("Balanced Recall (Class 0)", "87%")
    col3.metric("Regression MAE", "0.62")

    st.markdown("---")

    st.markdown("""
    **Models Compared:**
    - Logistic Regression (Baseline)
    - Balanced Logistic Regression
    - Random Forest Classifier
    - Linear Regression
    
    Random Forest achieved highest classification performance.
    """)

# ------------------------------
# LIVE PREDICTION
# ------------------------------

user_input = st.text_area("Enter a customer review:")

if st.button("Predict Recommendation"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Convert text to TF-IDF features
        input_vector = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.success("Customer is likely to RECOMMEND this product ✅")
        else:
            st.error("Customer is NOT likely to recommend this product ❌")


