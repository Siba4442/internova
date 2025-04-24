import streamlit as st
import PyPDF2
import spacy
import os
# from spacy.matcher import Matcher
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
def app():
    # Initialize resources once
    if 'initialized' not in st.session_state:
        model_path = os.path.join(os.path.dirname(__file__), 'models/en_core_web_sm-3.7.1')
        st.session_state['nlp'] = spacy.load(model_path)

        # List of skills
        with open('Skills.json', 'r') as f:
            skills = json.load(f)

        with open('SurfaceFrom.json', 'r') as f:
            SurfaceFrom = json.load(f)
        
        with open('Skills_ID.json', 'r') as f:
            Skills_ID = json.load(f)
            
        with open('SurfaceFrom_ID.json', 'r') as f:
            SurfaceFrom_ID = json.load(f)

        # Load dataset and Word2Vec model
        file_name = 'final_2nd.csv'
        file_path = os.path.join(os.getcwd(), file_name)
        df = pd.read_csv(file_path)
        job_skills = df['skills']
        skills_processed = [item.lower().replace(',', ' ').replace(' ', '_').split() for item in skills]
        word2vec_model = Word2Vec(skills_processed, vector_size=100, window=5, min_count=1)

        st.session_state.update({
            'job_skills': job_skills,
            'df': df,
            'word2vec_model': word2vec_model,
            'SurfaceForm' : SurfaceFrom,
            'SurfaceForm_ID' : SurfaceFrom_ID,
            'Skills_ID' : Skills_ID,
            'initialized': True
        })


    # Skill extraction using matcher
    def skills_extract(text, threshold = 0.9):
        doc = st.session_state['nlp'](text)
        phrases = [chunk.text for chunk in doc.noun_chunks]
        combined_texts = phrases + st.session_state['SurfaceForm']
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_texts)
        phrase_vectors = tfidf_matrix[:len(phrases)]
        surface_word_vectors = tfidf_matrix[len(phrases):]
        mtch_list = []
        for i, phrase in enumerate(phrases):
        # Compute cosine similarity between the current phrase and all surface words
            similarities = cosine_similarity(phrase_vectors[i], surface_word_vectors).flatten()
            
            # Find the best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score >= threshold:
                best_surface_word = st.session_state['SurfaceForm'][best_idx]
                skill_id = st.session_state['SurfaceForm_ID'][best_surface_word]
                skill = st.session_state['Skills_ID'][skill_id]
                mtch_list.append(skill)
        
        return mtch_list

    # Compute vector for skills list
    def compute_vector(words_list):
        model = st.session_state['word2vec_model'].wv
        vectors = [model[word] for word in words_list if word in model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


    # Calculate similarity between user skills and job skills
    def calculate_similarity(user_skills):
        user_vector = compute_vector([skill.lower().replace(' ', '_') for skill in user_skills]).reshape(1, -1)
        return [
            cosine_similarity(user_vector, compute_vector([skill.lower().replace(' ', '_') for skill in job_skill]).reshape(1, -1))[0][0]
            for job_skill in st.session_state['job_skills']
        ]


    # Generate job recommendations based on similarity threshold and filters
    def generate_recommendations(user_data, location_type=None, job_location=None, threshold=0):
        recommendations = []
        df = st.session_state['df']

        for user_skills in user_data['skillsofusers']:
            sim_scores = calculate_similarity(user_skills)
            if len(sim_scores) != len(df):
                st.write(f"Length mismatch: len(sim_scores) = {len(sim_scores)}, len(df) = {len(df)}")
        
            df_copy = df.copy()
            if len(sim_scores) == len(df_copy):
                df_copy['similarity'] = sim_scores
            else:
                st.write("Mismatch in length of similarity scores and DataFrame")
                continue
        
            if location_type:
                df_copy = df_copy[df_copy['location_type'] == location_type]
            if job_location:
                df_copy = df_copy[df_copy['job_location'] == job_location]
        
            filtered_results = df_copy[df_copy['similarity'] >= threshold].sort_values('similarity', ascending=False)
        
            if filtered_results.empty:
                st.write(f"No job recommendations found for the selected filters (Location Type: {location_type}, Job Location: {job_location}).")
            else:
                recommendations.append(filtered_results)

        return recommendations


    # Streamlit UI
    st.title("INTERNOVA - Candidate Hub")
    st.write("### Discover the Perfect Opportunity for Your Career")


    # DataFrame for user inputs
    users_df1 = pd.DataFrame(columns=['userNames', 'skillsofusers'])
    skills_users, users_names = [], []

    # PDF uploader and skill extraction
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = " ".join(page.extract_text() for page in pdf_reader.pages)
        extracted_skills = skills_extract(pdf_text)
        st.text_area("Extracted Skills", ', '.join(extracted_skills), height=200)

    # Handle NaN values in job_location
    domestic_jobs_df = st.session_state['df'][st.session_state['df']['location_type'] == 'Domestic']
    job_location_unique_domestic = domestic_jobs_df['job_location'].fillna('Unknown').unique()
    job_location_sorted = sorted(job_location_unique_domestic)

    # Location filters
    location_type = st.selectbox("Select Location Type", ["Any", "Domestic", "International"])

    if location_type == "Domestic":
        job_location = st.selectbox("Select Job Location", ["Any"] + job_location_sorted)
    else:
        job_location = None

    # Submit button to generate recommendations
    if st.button("Submit"):
        user_id = len(users_names) + 1
        users_names.append(user_id)
        skills_users.append(extracted_skills)

        if users_names:
            users_df1['userNames'], users_df1['skillsofusers'] = users_names, skills_users
            recommend_list = generate_recommendations(users_df1, location_type if location_type != "Any" else None, job_location if job_location != "Any" else None)
            if not recommend_list:
                st.write(f"Pleaae try dffferent location")
            else:
                st.write(f"### Top 5 Job Recommendations")
                for user_idx, top_recommendations in enumerate(recommend_list):
                    for index, row in top_recommendations.head(5).iterrows():
                        with st.expander(f"**{row['job_title']} - {row['company']}**", expanded=False):
                            st.write(f"*Company:* {row['company']}")
                            st.write(f"*Type:* {row['employment_type']}")
                            st.write(f"*Location:* {row['job_location']}")
                            st.markdown(f"*Apply now:* [Click here]({row['job_link']})")
        else:
            st.write("Please upload a PDF with valid data before submitting.")
        
        
        
if __name__ == "__main__":
    app()