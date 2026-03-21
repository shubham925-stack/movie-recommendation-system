import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv("dataset/movies.csv")

# Load ratings dataset
ratings = pd.read_csv("dataset/ratings.csv")

print("Movies Dataset:\n")
print(movies.head())

print("\nRatings Dataset:\n")
print(ratings.head())

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

print("\nMerged Dataset:\n")
print(data.head())

# Movie rating statistics
movie_ratings = data.groupby("title").agg({"rating": ["mean", "count"]})

# Rename columns
movie_ratings.columns = ["average_rating", "rating_count"]

print("\nMovie Rating Statistics:\n")
print(movie_ratings.head())

# Filter movies with at least 50 ratings
popular_movies = movie_ratings[movie_ratings["rating_count"] >= 50]

print("\nPopular Movies:\n")
print(popular_movies.head())

# Create user–movie matrix
user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating")

# Keep only popular movies
user_movie_matrix = user_movie_matrix[popular_movies.index]

print("\nUser Movie Matrix:\n")
print(user_movie_matrix.head())

# Fill missing values
movie_matrix_filled = user_movie_matrix.fillna(0)

# Calculate movie similarity
movie_similarity = cosine_similarity(movie_matrix_filled.T)

# Convert similarity matrix to DataFrame
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=movie_matrix_filled.columns,
    columns=movie_matrix_filled.columns
)

print("\nMovie Similarity Matrix:\n")
print(movie_similarity_df.head())


# ---------------------------------------------------
# Smart movie search function
# ---------------------------------------------------
def search_movie(movie_name):

    movie_name = movie_name.lower()

    for title in movie_similarity_df.columns:
        if movie_name in title.lower():
            return title

    return None


# ---------------------------------------------------
# Recommendation function
# ---------------------------------------------------
def recommend_movies(movie_name, num_recommendations=5):

    if movie_name not in movie_similarity_df.columns:
        print("Movie not found in dataset.")
        return None

    similar_scores = movie_similarity_df[movie_name]

    similar_movies = similar_scores.sort_values(ascending=False)

    return similar_movies.iloc[1:num_recommendations+1]


# ---------------------------------------------------
# Interactive recommendation system
# ---------------------------------------------------
print("\n===== Movie Recommendation System =====")

user_input = input("Enter a movie name: ")

movie_name = search_movie(user_input)

if movie_name is None:
    print("Movie not found in dataset.")
else:
    recommendations = recommend_movies(movie_name)

    print(f"\nRecommendations based on: {movie_name}\n")

    for movie in recommendations.index:
        print(movie)

import pickle

pickle.dump(movies.to_dict(), open('movies.pkl', 'wb'))
pickle.dump(movie_similarity, open('similarity.pkl', 'wb'))