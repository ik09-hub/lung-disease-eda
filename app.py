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

def plot_count(df, feature):
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x=feature, order=df[feature].value_counts().index, ax=ax1)
    ax1.set_title(f"{feature} Distribution")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

def plot_count_by_recovered(df, feature):
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x=feature, hue="Recovered", order=df[feature].value_counts().index, ax=ax2)
    ax2.set_title(f"{feature} vs Recovery")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
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

    with st.expander("1. Age"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Age Distribution")
            st.image("output/age_distribution.png")
        with col2:
            st.subheader("Age vs Recovery")
            st.image("output/age_distribution_by_recovered.png")       
        st.subheader("Age Groups Binned")
        plot_count_by_recovered(df, "Age Group")
    
    with st.expander("2. Gender"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gender Distribution")
            plot_count(df, "Gender")
        with col2:
            st.subheader("Gender vs Recovery")
            plot_count_by_recovered (df, "Gender")
    with st.expander("3. Smoking Status"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Smoking Status Distribution")
            plot_count(df, "Smoking Status")
        with col2:
            st.subheader("Smoking Status vs Recovery")
            plot_count_by_recovered (df, "Smoking Status")
    with st.expander("4. Disease Type"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Disease Type Distribution")
            plot_count(df, "Disease Type")
        with col2:
            st.subheader("Disease Type vs Recovery")
            plot_count_by_recovered (df, "Disease Type")
    with st.expander("5. Treatment Type"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Treatment Type Distribution")
            plot_count(df, "Treatment Type")
        with col2:
            st.subheader("Treatment Type vs Recovery")
            plot_count_by_recovered (df, "Treatment Type")      
    with st.expander("6. Hospital Visits"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hospital Visits Distribution")
            plot_count(df, "Hospital Visits")
        with col2:
            st.subheader("Hospital Visits vs Recovery")
            plot_count_by_recovered (df, "Hospital Visits")  
            
        st.subheader("Hospital Visits Binned")
        plot_count_by_recovered(df, "Visit_Tier")

    with st.expander("7. Lung Capacity"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lung Capacity Distribution")
            plot_count(df, "Lung Capacity")
        with col2:
            st.subheader("Lung Capacity vs Recovery")
            plot_count_by_recovered (df, "Lung Capacity")  
            
        st.subheader("Lung Capacity Binned")
        plot_count_by_recovered(df, "LungCapacity_Level")



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

