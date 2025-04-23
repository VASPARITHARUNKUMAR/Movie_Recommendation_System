import streamlit as st
import pandas as pd
import requests
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ========== Page Config ==========
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎥 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ========== Load Data ==========
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

movies = pd.read_pickle('processed_movies.pkl')

# ========== Helper: Fetch Poster ==========
def fetch_poster(movie_id):
    api_key = '66025b3fe3b6327594888459ecea084e'  # Replace with your own key
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    response = requests.get(url)
    data = response.json()
    if 'poster_path' in data and data['poster_path']:
        return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    else:
        return "https://via.placeholder.com/200x300?text=No+Image"

# ========== Helper: Recommend Movies ==========
def recommend(movie_title, top_n=10):
    if movie_title not in movies['title'].values:
        return pd.DataFrame()

    index = movies[movies['title'] == movie_title].index[0]
    vector = tfidf_matrix[index].reshape(1, -1)
    similarity_scores = cosine_similarity(vector, tfidf_matrix)[0]

    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
    results = movies.iloc[similar_indices][['title', 'movie_id']].copy()
    results['score'] = similarity_scores[similar_indices]
    return results

# ========== User Input ==========
with st.container():
    selected_movie = st.selectbox(
        "🎬 Pick a movie you like:",
        sorted(movies['title'].unique()),
        index=sorted(movies['title'].unique()).index("The Dark Knight") if "The Dark Knight" in movies['title'].values else 0
    )

    st.markdown("🔍 Click the button below to discover similar movies based on content and genre.")

    if st.button("✨ Recommend"):
        recommendations = recommend(selected_movie)

        if recommendations.empty:
            st.warning("No recommendations found for this movie.")
        else:
            st.markdown("<h3 style='text-align: center;'>📢 Top Recommendations for <em>{}</em></h3>".format(selected_movie), unsafe_allow_html=True)
            for i in range(0, 10, 5):  # 2 rows of 5 posters
                cols = st.columns(5)
                for col, j in zip(cols, range(i, i+5)):
                    if j < len(recommendations):
                        title = recommendations.iloc[j]['title']
                        movie_id = recommendations.iloc[j]['movie_id']
                        poster_url = fetch_poster(movie_id)
                        with col:
                            st.image(poster_url, use_column_width=True)
                            st.markdown(f"<div style='text-align: center; font-weight: bold;'>{title}</div>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Developed with ❤️ using Streamlit</div>", unsafe_allow_html=True)
