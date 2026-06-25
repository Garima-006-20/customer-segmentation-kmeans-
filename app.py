import streamlit as st
import joblib
import numpy as np

# ------------------ PAGE TITLE ------------------
st.set_page_config(
    page_title="Customer Segmentation using K-Means",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Customer Segmentation using K-Means Clustering")

# ------------------ SIDEBAR ------------------
st.sidebar.header("📌 About Project")

st.sidebar.info("""
This application uses the K-Means Clustering algorithm to segment customers based on their spending behavior.

It helps businesses identify different customer groups and create targeted marketing strategies.
""")

st.sidebar.header("👩‍💻 Developer Information")
st.sidebar.write("**Name:** Garima")
st.sidebar.write("**Project:** Customer Segmentation using K-Means")


# ------------------ PROJECT DESCRIPTION ------------------
st.markdown("""
## 📖 Project Description

Customer segmentation is a crucial business strategy used to understand customer behavior and purchasing patterns. Instead of treating all customers similarly, organizations can divide customers into meaningful groups and design personalized marketing campaigns.

This project implements the **K-Means Clustering Algorithm**, an unsupervised machine learning technique, to classify customers based on their characteristics.

The segmentation is performed using the following customer attributes:

- 💰 **Annual Income (k$)**
- 🛒 **Spending Score (1–100)**

By identifying different customer groups, businesses can:

- Improve customer satisfaction.
- Design targeted promotional campaigns.
- Identify premium and loyal customers.
- Increase profitability.
- Enhance decision-making through data-driven insights.

### 🎯 Objectives

- Analyze customer purchasing behavior.
- Discover hidden patterns within customer data.
- Segment customers into meaningful groups.
- Support personalized marketing strategies.
- Provide an interactive prediction system.

### ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

### 🤖 Machine Learning Technique

The project uses the **K-Means Clustering Algorithm**. The optimal number of clusters was determined using the **Elbow Method**, and clustering quality was evaluated using the **Silhouette Score**.

### 📂 Dataset Information

- Dataset: Mall Customers Dataset
- Total Customers: 200
- Features Used:
    - Annual Income (k$)
    - Spending Score (1–100)
""")
st.subheader("🔄 Methodology")

st.markdown("""
1. Data Collection (Mall Customers Dataset)
2. Data Preprocessing
3. Feature Selection
4. Feature Scaling using StandardScaler
5. Determining Optimal Clusters using Elbow Method
6. Customer Segmentation using K-Means
7. Evaluation using Silhouette Score
8. Streamlit Deployment
""")
st.metric("Silhouette Score", "0.55")
st.subheader("📊 Customer Segmentation Visualization")


# ------------------ LOAD MODEL ------------------
try:
    model = joblib.load("model/kmeans_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ------------------ USER INPUT ------------------
st.header("🔍 Predict Customer Segment")

income = st.number_input(
    "Enter Annual Income (k$)",
    min_value=0,
    max_value=150,
    value=50
)

spending = st.number_input(
    "Enter Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

# ------------------ PREDICTION ------------------
if st.button("Predict Segment"):

    customer = np.array([[income, spending]])

    customer_scaled = scaler.transform(customer)

    cluster = model.predict(customer_scaled)
    st.write("Predicted Cluster Number:", cluster[0])

    cluster_number = int(cluster[0])

    cluster_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "Impulsive Customers",
    3: "Careful Customers",
    4: "Budget Customers"
    }

    st.success(
        f"Predicted Segment: {cluster_names.get(cluster_number, f'Cluster {cluster_number}')}"
    )

    st.info(
        f"The customer belongs to the **{cluster_names.get(cluster_number)}** category."
    )

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "<center>Developed by <b>Garima</b> as part of Internship Project</center>",
    unsafe_allow_html=True
)
