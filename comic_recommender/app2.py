import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# Load your dataset
df = pickle.load(open("comics.pkl", "rb"))

# TF-IDF and similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create index mapping
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

# Recommender function
def recommend_comics(title, num_recommendations=5):
    if title not in indices:
        return None

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    comic_indices = [i[0] for i in sim_scores]
    return df[['Title', 'Author', 'genre', 'Rating','completion_status']].iloc[comic_indices].reset_index(drop=True)

# Streamlit UI
st.set_page_config(page_title="Comic Recommender", layout="centered")
st.title("📚 Comic Recommendation System")

# Input box
comic_title = st.selectbox("Select a Comic Title:", sorted(df['Title'].unique()))

if st.button("Get Recommendations"):
    results = recommend_comics(comic_title)

    if results is None:
        st.error(f"Comic '{comic_title}' not found.")
    else:
        st.subheader("Recommended Comics:")
        for i, row in results.iterrows():
            st.markdown(f"### {row['Title']}")
            st.markdown(f"- **Author:** {row['Author']}")
            st.markdown(f"- **Genre:** {row['genre']}")
            st.markdown(f"- **Rating:** {row['Rating']}")
            st.markdown("---")
