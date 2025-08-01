# 🛍️ Mall Customer Segmentation Web App

A user-friendly machine learning application developed with **Streamlit**, designed to segment mall customers by analyzing their **Annual Income** and **Spending Score** using **K-Means** and **DBSCAN** clustering algorithms.

---

📌 **Project Summary**

Customer segmentation plays a vital role in understanding consumer behavior and creating effective, targeted marketing campaigns. This web app offers an interactive platform to explore customer groupings based on income and spending patterns.

With a simple CSV file upload, the app allows you to:

* Visualize how customers are grouped into clusters
* Experiment with varying numbers of clusters
* Utilize density-based clustering via DBSCAN
* Extract insights based on spending trends in each cluster

---

🧠 **Tech Stack & Libraries**

* **Python**
* **Streamlit**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

---

⚙️ **Key Features**

* Upload and preview dataset from a CSV file
* Perform **K-Means** clustering with adjustable cluster count (K)
* Apply **DBSCAN** with customizable parameters (`eps` and `min_samples`)
* Visualize clusters through scatter plots
* Analyze average spending per cluster with bar charts
* Use an interactive sidebar for quick parameter tuning

---

📦 **How to Set It Up**

1. Download the project and open in vs code:



2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Run the application:

   ```
   streamlit run app.py
   ```
