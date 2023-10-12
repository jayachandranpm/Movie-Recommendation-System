from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)

# Path to your CSV file containing movie metadata
csv_file_path = 'dataset/mymoviedb.csv'

# Load movie metadata from the CSV file
metadata = pd.read_csv(csv_file_path, low_memory=False)

# Clean the data and handle missing values
metadata['title'] = metadata['title'].fillna('')
metadata['overview'] = metadata['overview'].fillna('')

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(metadata['overview'])

# Perform Truncated SVD to reduce dimensionality
n_components = 100  # Adjust the number of components as needed
svd = TruncatedSVD(n_components=n_components)
tfidf_matrix_svd = svd.fit_transform(tfidf_matrix)

# Calculate cosine similarity between movies using the reduced-dimensional TF-IDF matrix
cosine_sim = linear_kernel(tfidf_matrix_svd, tfidf_matrix_svd)

# Function to get top-N movie recommendations for a movie title
def get_top_n_recommendations(movie_title, n=10):
    # Convert the user input to title case
    movie_title = movie_title.title()

    movie_index = metadata[metadata['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_n_recommendations = similarity_scores[1:n+1]  # Exclude the movie itself
    top_n_movie_indices = [index for index, _ in top_n_recommendations]
    top_n_movie_titles = metadata['title'].iloc[top_n_movie_indices].tolist()
    return top_n_movie_titles

# Define a route for the web interface
@app.route('/', methods=['GET', 'POST'])
def movie_recommendation():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        top_recommendations = get_top_n_recommendations(movie_title)
        return render_template('index.html', movie_title=movie_title, recommendations=top_recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
