import streamlit as st
import pandas as pd
import joblib
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# --- 1. Load Pre-trained Models and Data ---

scaler = None
kmeans_final = None
sim_df = None

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('kmeans_final.pkl', 'rb') as f:
        kmeans_final = pickle.load(f)
    sim_df = joblib.load('item_similarity_matrix.joblib')
except FileNotFoundError:
    st.error("Error: Model files not found. Make sure 'scaler.pkl', 'kmeans_final.pkl', and 'item_similarity_matrix.joblib' are in the correct directory.")
    st.stop()

# Define the latest date from the original dataset for Recency calculation
# This should be the max(InvoiceDate) + 1 day from your training data
# For example, if max InvoiceDate was 2011-12-09, set this to 2011-12-10
latest_date = datetime(2011, 12, 9)

# Define the segment labels
segment_labels = {
    2: 'High-Value/Champions',
    3: 'Loyal / Regular',
    0: 'Occasional',
    1: 'At-Risk/Dormant'
}

# --- 2. RFM Segmentation Function ---
def segment_customer(recency, frequency, monetary):
    if scaler is None or kmeans_final is None:
        return "Models not loaded. Cannot segment customer."

    # Create a DataFrame for the new customer's RFM values
    new_customer_rfm = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])

    # Scale the new customer's RFM values
    scaled_rfm = scaler.transform(new_customer_rfm)

    # Predict the cluster
    cluster = kmeans_final.predict(scaled_rfm)[0]

    # Map the cluster to a segment label
    segment = segment_labels.get(cluster, "Unknown Segment")

    return segment

# --- 3. Item Recommendation Function ---
def recommend_products(product_description, num_recommendations=5):
    if sim_df is None:
        return "Similarity matrix not loaded. Cannot provide recommendations."

    if product_description not in sim_df.index:
        return f"Product '{product_description}' not found in our catalog. Please try an exact match or another product."

    # Get similarity scores for the product
    sims = sim_df[product_description].sort_values(ascending=False)

    # Get top N similar products (excluding itself)
    top_recommendations = sims.index[1:num_recommendations + 1]

    # Format output with similarity scores
    recommendations_with_scores = []
    for prod in top_recommendations:
        recommendations_with_scores.append(f"{prod} (Similarity: {sims[prod]:.3f})")

    return recommendations_with_scores

# --- 4. Streamlit Application Layout ---
st.set_page_config(page_title="Customer Segmentation & Product Recommendation", layout="wide")

st.title("🛒 Customer Segmentation & Product Recommendation")
st.markdown("Welcome to our e-commerce analytics dashboard. Use the tools below to understand your customers and suggest products.")

st.sidebar.header("Navigation")
selection = st.sidebar.radio("Go to", ["Customer Segmentation", "Product Recommendation"])

if selection == "Customer Segmentation":
    st.header("📊 Customer Segmentation (RFM)")
    st.markdown("Enter a customer's Recency, Frequency, and Monetary values to determine their segment.")

    # Input fields for RFM
    col1, col2, col3 = st.columns(3)
    with col1:
        recency_input = st.number_input("Recency (Days since last purchase)", min_value=0, value=30, step=1)
    with col2:
        frequency_input = st.number_input("Frequency (Number of purchases)", min_value=0, value=5, step=1)
    with col3:
        monetary_input = st.number_input("Monetary (Total spend)", min_value=0.0, value=250.0, step=0.1)

    if st.button("Segment Customer"):
        if recency_input is not None and frequency_input is not None and monetary_input is not None:
            segment = segment_customer(recency_input, frequency_input, monetary_input)
            st.success(f"This customer belongs to the: **{segment}** segment.")
        else:
            st.warning("Please fill in all RFM values.")

elif selection == "Product Recommendation":
    st.header("🛍️ Item-Based Product Recommendation")
    st.markdown("Get product recommendations based on a product you're interested in.")

    # Input for product description
    product_desc_input = st.text_input("Enter Product Description (Exact Match)", "RED WOOLLY HOTTIE WHITE HEART.")
    num_recs = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    if st.button("Get Recommendations"):
        if product_desc_input:
            recommendations = recommend_products(product_desc_input, num_recs)
            if isinstance(recommendations, list):
                st.subheader("Top Recommended Products:")
                for i, rec in enumerate(recommendations):
                    st.write(f"{i+1}. {rec}")
            else:
                st.warning(recommendations)
        else:
            st.warning("Please enter a product description.")

st.markdown("--- ")
st.write("Developed using Streamlit | Based on Online Retail Dataset")
