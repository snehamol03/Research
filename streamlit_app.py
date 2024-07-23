import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import re

# Preprocess the text
def preprocess_text_simple(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

# Load the model, vectorizer, and label encoder
with open('research_paper_categorization_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Streamlit UI
st.title('Research Paper Categorization')

# Input text
text = st.text_area("Enter the research paper summary:")

if st.button("Predict"):
    if text:
        # Preprocess and vectorize the text
        processed_text = preprocess_text_simple(text)
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict category
        predicted_category = model.predict(text_vectorized)
        category_label = label_encoder.inverse_transform(predicted_category)
        
        # Display result
        st.write(f"Predicted category: {category_label[0]}")
    else:
        st.write("Please enter a summary to predict.")