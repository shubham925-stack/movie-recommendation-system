import streamlit as st
import pickle
import pandas as pd
import os

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_dict = pickle.load(open(os.path.join(BASE_DIR, "movies.pkl"), 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open(os.path.join(BASE_DIR, "similarity.pkl"), 'rb'))
# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    # Safety check
    if movie_index >= len(similarity):
        return ["No recommendations available for this movie"]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# Streamlit UI
st.title('🎬 Movie Recommendation System')

selected_movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button('Recommend'):

    names = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.success(names[0])

    with col2:
        st.success(names[1])

    with col3:
        st.success(names[2])

    with col4:
        st.success(names[3])

    with col5:
        st.success(names[4])
    
    


    with col5:
        st.success(names[4])
        
