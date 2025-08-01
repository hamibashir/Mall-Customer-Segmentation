import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# Set page config
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

# Title
st.title("🛍️ Mall Customer Segmentation")

# File Upload
uploaded_file = st.file_uploader("Upload Mall Customer CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # Feature selection
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.sidebar.subheader("🔢 Select number of clusters (K)")
    k = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=5)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("📊 Clustered Data Preview")
    st.dataframe(df[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']])

    # Cluster scatterplot
    st.subheader("📈 Cluster Visualization")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', ax=ax1)
    st.pyplot(fig1)

    # Average spending barplot
    st.subheader("💸 Average Spending Score per Cluster")
    fig2, ax2 = plt.subplots()
    df.groupby('Cluster')['Spending Score (1-100)'].mean().plot(kind='bar', color='skyblue', ax=ax2)
    ax2.set_ylabel("Average Spending Score")
    st.pyplot(fig2)

    # Optional DBSCAN
if st.sidebar.checkbox("🔍 Try DBSCAN"):
    eps = st.sidebar.slider("DBSCAN eps (neighborhood size)", 0.1, 2.0, 0.5, step=0.1)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 10, 5)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

    st.subheader("🧪 DBSCAN Clustering")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='DBSCAN_Cluster', palette='Set1', ax=ax3)
    st.pyplot(fig3)

    unique_clusters = df['DBSCAN_Cluster'].nunique()
    st.write(f"👀 DBSCAN found {unique_clusters} cluster(s) including noise (-1).")

    # Average spending barplot
    st.subheader("💸 Average Spending Score per DBSCAN Cluster")
    fig4, ax4 = plt.subplots()
    df.groupby('DBSCAN_Cluster')['Spending Score (1-100)'].mean().plot(kind='bar', color='skyblue', ax=ax4)
    ax4.set_ylabel("Average Spending Score")
    st.pyplot(fig4)