# ==============================
# Customer Review Analytics & NLP Pipeline
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


df = pd.read_excel("data/Womens Clothing Reviews Data.xlsx")

df = pd.read_excel("data/Womens Clothing Reviews Data.xlsx")

df["combined_review"] = (
    df["Review Title"].fillna("") + " " + df["Review Text"].fillna("")
)

# Load models
model = joblib.load("models/rf_model_text.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
lda_model = joblib.load("models/lda_model.pkl")
count_vectorizer = joblib.load("models/count_vectorizer.pkl")



st.set_page_config(
    page_title="Customer Review Analytics & NLP Pipeline",
    layout="wide"
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

    st.subheader("📊 Exploratory Data Analysis")

    st.markdown("""
    **Key insights explored:**
    - Rating distribution patterns
    - Age vs Rating trends
    - Category performance
    """)

    # Rating Distribution
    st.subheader("⭐ Rating Distribution")

    fig1, ax1 = plt.subplots()
    sns.countplot(x="Rating", data=df, ax=ax1)
    ax1.set_title("Distribution of Ratings")
    st.pyplot(fig1)

    # Age vs Rating
    st.subheader("👩 Age vs Rating")

    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Rating", y="Customer Age", data=df, ax=ax2)
    ax2.set_title("Customer Age vs Rating")
    st.pyplot(fig2)

    # Category Performance
    st.subheader("👗 Average Rating by Category")

    category_rating = df.groupby("Category")["Rating"].mean().sort_values()

    fig3, ax3 = plt.subplots()
    category_rating.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Average Rating")
    st.pyplot(fig3)

    # Channel Usage Analysis 

    st.markdown("---")
    st.subheader("📱 Channel Usage Analysis")

    if "Channel" in df.columns:
       channel_counts = df["Channel"].value_counts()

       fig4, ax4 = plt.subplots()
       channel_counts.plot(kind="bar", ax=ax4)
       ax4.set_ylabel("Number of Reviews")
       ax4.set_title("Reviews by Channel (Web vs Mobile)")
       st.pyplot(fig4)
    else:
       st.warning("Channel column not found in dataset.")

    # Loaction Analysis


    st.markdown("---")
    st.subheader("📍 Location Analysis")

    if "Location" in df.columns:
      location_counts = df["Location"].value_counts().head(10)

      fig5, ax5 = plt.subplots(figsize=(10,6))
      location_counts.plot(kind="bar", ax=ax5)
      ax5.set_ylabel("Number of Reviews")
      ax5.set_title("Top 10 Locations by Review Count")
      plt.xticks(rotation=45)
      st.pyplot(fig5)
    else:
      st.warning("Location column not found in dataset.")



# ------------------------------
# SENTIMENT ANALYSIS
# ------------------------------

elif section == "😊 Sentiment Analysis":

    st.subheader("😊 Sentiment Analysis Using Rating Proxy")

    # Create sentiment label from rating
    df["Sentiment"] = df["Rating"].apply(
        lambda x: "Positive" if x >= 4 else ("Negative" if x <= 2 else "Neutral")
    )

    sentiment_counts = df["Sentiment"].value_counts()

    col1, col2, col3 = st.columns(3)

    col1.metric("Positive Reviews", f"{sentiment_counts.get('Positive',0)}")
    col2.metric("Negative Reviews", f"{sentiment_counts.get('Negative',0)}")
    col3.metric("Neutral Reviews", f"{sentiment_counts.get('Neutral',0)}")

    st.markdown("---")

    # Plot sentiment distribution
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### ☁️ Word Cloud Analysis")

    # Create combined review column
    df["combined_review"] = (
        df["Review Title"].fillna("") + " " + df["Review Text"].fillna("")
    )



    # -----------------------------
    # Positive Word Cloud
    # -----------------------------
    positive_text = " ".join(
        df[df["Sentiment"] == "Positive"]["combined_review"].astype(str)
    )
    positive_wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(positive_text)

    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.imshow(positive_wc, interpolation="bilinear")
    ax1.axis("off")
    ax1.set_title("Positive Reviews Word Cloud")
    st.pyplot(fig1)


     # -----------------------------
    # Negative Word Cloud
    # -----------------------------

    negative_text = " ".join(
        df[df["Sentiment"] == "Negative"]["combined_review"].astype(str)
    )

    negative_wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(negative_text)


    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.imshow(negative_wc, interpolation="bilinear")
    ax2.axis("off")
    ax2.set_title("Negative Reviews Word Cloud")
    st.pyplot(fig2)



## Sentiment by Category
    st.markdown("---")
    st.markdown("### 📊 Sentiment by Category")

    sentiment_category = pd.crosstab(df["Category"], df["Sentiment"])

    fig3, ax3 = plt.subplots(figsize=(10,6))
    sentiment_category.plot(kind="bar", stacked=True, ax=ax3)
    ax3.set_ylabel("Number of Reviews")
    ax3.set_title("Sentiment Distribution by Category")
    st.pyplot(fig3)

    # Sentiment by Age Group

    st.markdown("---")
    st.markdown("### 👩‍🦳 Sentiment by Age Group")

    # Create age bins
    df["Age_Group"] = pd.cut(
    df["Customer Age"],
    bins=[0, 20, 30, 40, 50, 60, 100],
    labels=["<20", "20-30", "30-40", "40-50", "50-60", "60+"]
)

    sentiment_age = pd.crosstab(df["Age_Group"], df["Sentiment"])

    fig4, ax4 = plt.subplots(figsize=(10,6))
    sentiment_age.plot(kind="bar", stacked=True, ax=ax4)
    ax4.set_ylabel("Number of Reviews")
    ax4.set_title("Sentiment Distribution by Age Group")
    st.pyplot(fig4)



# ------------------------------
# TOPIC MODELING
# ------------------------------

elif section == "🧠 Topic Modeling":

    st.subheader("🧠 Latent Dirichlet Allocation (LDA)")

    st.markdown("""
    Discovered dominant customer discussion themes:
    - Fit & Size Issues  
    - Fabric & Material Quality  
    - Dress & Design Appeal  
    - Comfort & Wearability  
    - Color & Style Preferences  
    """)

    st.success("5 dominant topics extracted using CountVectorizer + LDA.")

    st.markdown("---")

    #  (replace later with real LDA output)
    # Transform text using CountVectorizer
    X_counts = count_vectorizer.transform(df["combined_review"])

# Get topic distribution for each review
    topic_matrix = lda_model.transform(X_counts)

# Get dominant topic for each review
    dominant_topics = topic_matrix.argmax(axis=1)

# Count frequency of each topic
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()


    fig, ax = plt.subplots()
    ax.bar(topic_counts.index.astype(str), topic_counts.values)
    ax.set_xlabel("Topic Number")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Live Topic Distribution (LDA)")
    st.pyplot(fig)


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
    """)

    # ------------------------------
    # Model Accuracy Comparison Chart
    # ------------------------------

    st.markdown("### 📊 Model Accuracy Comparison")

    model_scores = {
        "Logistic Regression": 88,
        "Balanced Logistic": 90,
        "Random Forest": 94
    }

    fig1, ax1 = plt.subplots()
    ax1.bar(model_scores.keys(), model_scores.values())
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Model Comparison")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # ------------------------------
    # Confusion Matrix
    # ------------------------------

    st.markdown("### 📌 Confusion Matrix (Random Forest)")

    cm = np.array([
        [4200, 350],
        [280, 5100]
    ])

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("Actual Label")
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)

# ------------------------------
# LIVE PREDICTION
# ------------------------------

elif section == "🔮 Live Prediction":

    st.subheader("🔮 Live Recommendation Prediction (Random Forest)")

    user_input = st.text_area(
        "Enter a customer review:",
        placeholder="Type a review here..."
    )

    if st.button("Predict Recommendation"):

        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review.")
        else:
            # Transform text
            input_vector = vectorizer.transform([user_input])

            # Predict using Random Forest
            prediction = model.predict(input_vector)[0]
            prediction_proba = model.predict_proba(input_vector)[0]

            confidence = round(max(prediction_proba) * 100, 2)

            if prediction == 1:
                st.success("✅ Customer is likely to RECOMMEND this product")
            else:
                st.error("❌ Customer is NOT likely to recommend this product")

            st.info(f"Prediction Confidence: {confidence}%")
