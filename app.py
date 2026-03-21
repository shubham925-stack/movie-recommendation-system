import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- SESSION STATE FOR LANDING PAGE ----------------   
if "page" not in st.session_state:
    st.session_state.page = "home"

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


# ---------------- LANDING PAGE ----------------
if st.session_state.page == "home":

    st.markdown(
        "<h1 style='text-align:center;'>🎬 Movie Recommendation System</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align:center;'>Discover movies similar to the ones you love</h3>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("🚀 Start Exploring"):
            st.session_state.page = "app"
            st.rerun()


# ---------------- YOUR ORIGINAL APP (UNCHANGED) ----------------
if st.session_state.page == "app":

    # Streamlit UI
    st.title('🎬 Movie Recommendation System')

    selected_movie_name = st.selectbox(
        "Select a movie",
        movies.iloc[:len(similarity)]['title'].values
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
     