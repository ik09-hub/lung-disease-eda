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
    st.title("🫁 Lung Disease Recovery Prediction")

    st.markdown("""
    Welcome to this interactive dashboard!  
    Here’s what you’ll find:

        - 📊 EDA — Explore distributions by age, gender, smoking status, etc.  
        - 🤖 Model Results — See how logistic regression and random forest perform  
        - 🌲 Feature Importance — Understand which features mattered most  
        - 🧠 Reflection — Honest analysis on model limitations and insights

    ---
    """)

    st.markdown("> *Can we predict who recovers from lung disease using basic clinical data?* This project explores that question using real-world patient information.")



# ---------------------
elif section == "📊 EDA":
    st.header("Exploratory Data Analysis")
    st.markdown("""    
    This section visualizes key features in the dataset to understand their distributions and relationships with recovery status.
    """)

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
            st.image("output/lung_capacity_distribution.png")
        with col2:
            st.subheader("Lung Capacity vs Recovery")
            st.image("output/lung_capacity_distribution_by_recovered.png")
            
        st.subheader("Lung Capacity Binned")
        plot_count_by_recovered(df, "LungCapacity_Level")

    st.markdown("""After exploring these features, we can see that: there is not a significant 
                difference in recovery rates across these different featues. This suggests a poor correlation
                between these features and the recovery outcome. The next step is to apply machine learning models to see if we can find any patterns.""")

# ---------------------
elif section == "🤖 Model Results":
    st.header("Model Results Summary")
    st.markdown("""
    This section summarizes the performance of the logistic regression and random forest 
    models applied to predict recovery from lung disease.
    """)
    st.markdown("""
    **Logistic Regression**
    - Accuracy: 0.55
    - Precision: 0.55
    - Recall: 0.58  
    - F1 Score: 0.56  
    - ROC AUC: 0.55  

    **Random Forest**  
    - Accuracy: 0.49
    - Precision: 0.49
    - Recall: 0.49
    - F1 Score: 0.49  
    - ROC AUC: 0.52  
    """)

    st.markdown(""" As seen above, both models performed similarly, with logistic regression slightly outperforming random forest.
    The ROC AUC scores indicate that both models are only slightly better than random guessing. Even despite implementing random forest 
                which is typically more robust to overfitting, the results are not promising.
    This suggests that the features in the dataset may not have a strong signal for predicting recovery.""")

    st.image("output/confusion_matrix_logreg.png", caption="Confusion Matrix: Logistic Regression")
    st.image("output/confusion_matrix_rf.png", caption="Confusion Matrix: Random Forest")

    st.markdown("""The confusion matrices show that both models struggle to correctly classify the positive class (recovered patients),
    with many false negatives. This indicates that the models are not effectively capturing the patterns needed to predict recovery accurately.""")
# ---------------------
elif section == "🌲 Feature Importance":
    st.header("Feature Importance (Random Forest)")

    st.markdown("""This section visualizes the feature importance scores from the random forest model.
    Feature importance helps us understand which features contribute most to the model's predictions. 
    From the random forest model, we can see that the most important features are lung capacity, age, and hospital visits.
    However, the overall performance of the model is still quite low, indicating that these features may not have a strong predictive power for recovery.""")
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
    
    Conclusions:
    After exploring the dataset and applying machine learning models, it is clear that the current features do not provide a strong signal for predicting recovery from lung disease.
    The models struggled to differentiate between recovered and non-recovered patients, resulting in low accuracy and AUC scores. Even though this project didn't yield a high predictive 
    performance, it highlights the importance of feature selection and the need for more informative data especially in medical datasets where feature prediction is very difficult. 
    """)

