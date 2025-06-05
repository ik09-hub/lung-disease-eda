import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Lung Disease Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/lung_disease_clean.csv")

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["🏠 Introduction", "📊 EDA", "🤖 Model Results", "🌲 Feature Importance", "🧠 Reflection"])

# ---------------------
if section == "🏠 Introduction":
    st.title("Lung Disease Recovery Prediction")
    st.markdown("""
    This dashboard explores a patient dataset to examine whether we can predict recovery from lung disease.
    I applied logistic regression and random forest models and evaluated their performance based on precision, recall, and AUC.
    """)

# ---------------------
elif section == "📊 EDA":
    st.header("Exploratory Data Analysis")
    st.subheader("Recovery Counts")
    st.bar_chart(df["Recovered"].value_counts())

    st.subheader("Recovery by Disease Type")
    st.bar_chart(df.groupby("Disease Type")["Recovered"].value_counts().unstack())

# ---------------------
elif section == "🤖 Model Results":
    st.header("Model Results Summary")
    st.markdown("""
    **Logistic Regression**  
    - F1 Score: 0.56  
    - ROC AUC: 0.55  

    **Random Forest**  
    - F1 Score: 0.49  
    - ROC AUC: 0.52  
    """)
    st.image("output/confusion_matrix_logreg.png", caption="Confusion Matrix: Logistic Regression")
    st.image("output/confusion_matrix_rf.png", caption="Confusion Matrix: Random Forest")

# ---------------------
elif section == "🌲 Feature Importance":
    st.header("Feature Importance (Random Forest)")
    st.image("output/feature_importance_rf.png")

# ---------------------
elif section == "🧠 Reflection":
    st.header("Reflection")
    st.markdown("""
    Despite a full preprocessing and modeling pipeline, both models performed only slightly better than random guessing.  
    Possible reasons include:
    - Weak signal in features
    - Overlap in patient characteristics
    - No symptom severity or lab values

    Future steps:
    - Try SMOTE for class balancing
    - Incorporate richer clinical features
    """)

