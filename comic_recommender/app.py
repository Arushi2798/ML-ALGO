import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load data
df = pickle.load(open("comics.pkl", "rb"))
pt = pickle.load(open("pt.pkl", "rb"))
similarity_scores = pickle.load(open("similarity.pkl", "rb"))

def recommend_comics(comic_title):
    if comic_title not in pt.index:
        return []

    index = np.where(pt.index == comic_title)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    recommendations = []
    for i in similar_items:
        comic_name = pt.index[i[0]]
        author = df[df['Title'] == comic_name]['Author'].iloc[0]
        genre = df[df['Title'] == comic_name]['genre'].iloc[0]
        recommendations.append((comic_name, author, genre))
    return recommendations

# Streamlit UI
st.title("📚 Comic Recommendation System")
comic_list = pt.index.tolist()
selected_comic = st.selectbox("Choose a Comic Title:", comic_list)

if st.button("Recommend"):
    recommendations = recommend_comics(selected_comic)
    if recommendations:
        st.subheader("Recommended Comics:")
        for name, author, genre in recommendations:
            st.markdown(f"**{name}**\n\n- *Author:* {author}\n- *Genre:* {genre}\n")
    else:
        st.warning("Comic not found or not enough data to recommend.")
