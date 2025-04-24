import streamlit as st
import pickle
import os
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Step 2: Preprocess Text (Lowercasing)
def preprocess_text(text):
    return text.lower()

# Step 3: Load Precomputed Embeddings
def load_precomputed_embeddings(load_dir="precomputed_embeddings"):
    with open(os.path.join(load_dir, "tfidf_vectorizer.pkl"), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    with open(os.path.join(load_dir, "tfidf_matrix.pkl"), 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    return tfidf_vectorizer, tfidf_matrix

# Step 4: Calculate ATS Score
def calculate_ats_score(resume_text, tfidf_vectorizer, tfidf_matrix, category_index):
    resume_text = preprocess_text(resume_text)
    
    # Transform the input resume into TF-IDF
    resume_tfidf = tfidf_vectorizer.transform([resume_text])
    
    # Calculate cosine similarity with the specific category
    similarity = cosine_similarity(resume_tfidf, tfidf_matrix[category_index])
    
    # Scale the similarity score to a percentage (0-100)
    ats_score = np.round(similarity[0] * 100, 2)
    return ats_score

# Streamlit Interface for ATS Checking
def app():
    st.title("ATS Score Checker")
    st.write("Select a job category and upload your resume in PDF format to get the ATS score.")

    # Dropdown menu for selecting job category
    categories = ["TEACHER", "SALES", "PUBLIC-RELATIONS", "INFORMATION-TECHNOLOGY", "HR",
                  "HEALTHCARE", "FITNESS", "FINANCE", "ENGINEERING", "DIGITAL-MEDIA",
                  "DESIGNER", "CONSULTANT", "CONSTRUCTION", "CHEF", "BUSINESS-DEVELOPMENT",
                  "BANKING", "AVIATION", "BPO", "AUTOMOBILE", "ARTS", "APPAREL",
                  "AGRICULTURE", "ADVOCATE", "ACCOUNTANT"]

    selected_category = st.selectbox("Select Job Category", categories)

    # Load precomputed embeddings
    tfidf_vectorizer, tfidf_matrix = load_precomputed_embeddings()

    # If a category is selected, display the uploader
    if selected_category:
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        
        # Get the index of the selected category
        category_index = categories.index(selected_category)

        # If resume is uploaded, calculate ATS score
        if uploaded_file is not None:
            # Extract text from the uploaded resume
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # Calculate ATS score for the selected category
            ats_score = calculate_ats_score(resume_text, tfidf_vectorizer, tfidf_matrix, category_index)
            
            # Display ATS score for the selected job category
            st.write(f"ATS Score for {selected_category} job: {ats_score}%")

if __name__ == "__main__":
    app()