import streamlit as st
import pandas as pd
import requests
import pickle

# Load models and data
with open('movies_with_clusters.pkl', 'rb') as f:
    movies = pd.read_pickle(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Function to fetch movie poster
def fetch_poster(movie_id):
    api_key = 'cd890fba108987eb0fa4a9554d958709'  # 🔑 Replace with your own API key
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    response = requests.get(url)
    data = response.json()
    if 'poster_path' in data and data['poster_path']:
        return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    else:
        return "https://via.placeholder.com/150"

# Recommend movies based on KMeans cluster
def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return []

    cluster_id = movies[movies['title'] == movie_title]['cluster'].values[0]
    recommended = movies[(movies['cluster'] == cluster_id) & (movies['title'] != movie_title)]
    return recommended[['title', 'movie_id']].head(10)

# Streamlit UI
st.title("🎥 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie:", sorted(movies['title'].unique()))

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.subheader("Top 10 recommended movies:")

    for i in range(0, 10, 5):  # 2 rows, 5 posters each
        cols = st.columns(5)
        for col, j in zip(cols, range(i, i+5)):
            if j < len(recommendations):
                title = recommendations.iloc[j]['title']
                movie_id = recommendations.iloc[j]['movie_id']
                poster_url = fetch_poster(movie_id)
                with col:
                    st.image(poster_url, width=130)
                    st.caption(title)
