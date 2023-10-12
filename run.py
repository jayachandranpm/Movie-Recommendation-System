'''# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Create a Flask app
app = Flask(__name__)

# Path to your CSV file containing movie metadata
csv_file_path = 'dataset/movies_metadata.csv'

# Load movie metadata from the CSV file
metadata = pd.read_csv(csv_file_path, low_memory=False)

# Clean the data and handle missing values
metadata['title'] = metadata['title'].fillna('')
metadata['overview'] = metadata['overview'].fillna('')

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(metadata['overview'])

# Calculate cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

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
    app.run(debug=True)'''

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Configure the SQLAlchemy database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/movies_data'
db = SQLAlchemy(app)

# Define the 'movies' table model
class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    overview = db.Column(db.Text)

# Initialize the TF-IDF vectorizer and cosine similarity matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = None
cosine_sim = None

# Function to fetch movie metadata
def get_movie_metadata():
    # Check if the TF-IDF data is already loaded
    global tfidf_matrix, cosine_sim
    if tfidf_matrix is None or cosine_sim is None:
        # Load movie data from the database
        metadata = Movie.query.all()
        
        if not metadata:
            return None  # No data in the database
        
        metadata_list = [{'title': movie.title, 'overview': movie.overview} for movie in metadata]
        
        # Create a DataFrame
        metadata = pd.DataFrame(metadata_list)
        
        # Clean the data and handle missing values
        metadata['title'] = metadata['title'].fillna('')
        metadata['overview'] = metadata['overview'].fillna('')
        
        # Initialize the TF-IDF vectorizer and cosine similarity matrix
        tfidf_matrix = tfidf_vectorizer.fit_transform(metadata['overview'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return metadata

@app.route('/', methods=['GET', 'POST'])
def movie_recommendation():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        metadata = get_movie_metadata()
        if metadata is None:
            return render_template('index.html', movie_title=movie_title, recommendations=None)
        
        top_recommendations = get_top_n_recommendations(movie_title, metadata, n=10)
        return render_template('index.html', movie_title=movie_title, recommendations=top_recommendations)
    return render_template('index.html')

def get_top_n_recommendations(movie_title, metadata, n=10):
    if metadata is None:
        return []  # No data available

    movie_title = movie_title.title()

    # Search for the movie in the loaded data
    movie_indices = metadata[metadata['title'] == movie_title].index
    if len(movie_indices) == 0:
        return []  # Movie not found

    movie_index = movie_indices[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the movie itself and return recommendations
    top_n_recommendations = similarity_scores[1:n+1]
    top_n_movie_indices = [index for index, _ in top_n_recommendations]
    top_n_movie_titles = metadata['title'].iloc[top_n_movie_indices].tolist()
    return top_n_movie_titles

if __name__ == '__main__':
    app.run(debug=True)
