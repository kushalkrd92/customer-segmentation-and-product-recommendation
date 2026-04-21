# Customer Segmentation & Product Recommendation System

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-FF9F00?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An interactive **Streamlit web application** that performs **RFM-based Customer Segmentation** using K-Means clustering and provides **Item-based Product Recommendations** using Cosine Similarity.

Built on the **Online Retail Dataset**, this project helps e-commerce businesses identify high-value customers and suggest relevant products to boost sales and customer retention.

---

## ✨ Features

- **Customer Segmentation Module**  
  Input Recency, Frequency & Monetary (RFM) values → instantly classify customers into **4 meaningful segments**:
  - High-Value / Champions
  - Loyal / Regular
  - Occasional
  - At-Risk / Dormant

- **Product Recommendation Module**  
  Enter any product description → get top similar products with similarity scores (Item-Based Collaborative Filtering).

- Clean, responsive sidebar navigation and user-friendly interface.

---

## 🛠 Technologies Used

- **Python**
- **Streamlit** – Web dashboard
- **Pandas** – Data handling
- **Scikit-learn** – K-Means clustering & Cosine Similarity
- **Joblib & Pickle** – Model serialization
- **Git LFS** – Large file handling

---

## 📁 Project Structure

```bash
├── Customer_Segmentation_&_Product_Recommendations_in_E_Commerce.ipynb  # Training pipeline
├── test.py                                      # Streamlit application
├── scaler.pkl                                   # Saved StandardScaler
├── kmeans_final.pkl                             # Trained K-Means model
├── item_similarity_matrix. joblib               # Item similarity matrix (LFS)
├── online_retail.csv                            # Original dataset (LFS)
├── README.md
├── .gitattributes                               # Git LFS configuration
└── requirements.txt                             # (optional) dependencies
