from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Define a fallback genre for unpredicted movies
fallback_genre = 'Unknown'

# List of Genres
genre_list = [ 'action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'family', 'fantasy', 'game-show', 'history', 'horror', 'music', 'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller', 'war', 'western' ]

# Load Training and Test Datasets (assuming CSV format)
train_data = pd.read_csv('train_data.txt', sep='>>', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
test_data = pd.read_csv('test_data.txt', sep='>>', header=None, names=['SerialNumber', 'MOVIE_NAME', 'MOVIE_PLOT'], engine='python')

# Ensure 'MOVIE_PLOT' is of string type
train_data['MOVIE_PLOT'] = train_data['MOVIE_PLOT'].astype(str)
test_data['MOVIE_PLOT'] = test_data['MOVIE_PLOT'].astype(str)

# Define MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Create a Text Preprocessing Function
def preprocess_text(text):
  return text.lower()  # Lowercasing

# Create a Machine Learning Pipeline
pipeline = Pipeline([
    ('text_preprocessing', Pipeline([('selector', lambda x: x['MOVIE_PLOT']), ('preprocessor', preprocess_text)])),
    ('tfidf_vectorizer', TfidfVectorizer(max_features=5000)),  # Adjust max_features
    ('multi_output_classifier', MultiOutputClassifier(MultinomialNB()))
])

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data['MOVIE_PLOT'], mlb.fit_transform([genre.split(', ') for genre in train_data['GENRE']]), test_size=0.2, random_state=42)

# Train the Model on Training Data
pipeline.fit(X_train, y_train)

# Predict Genres on Test Data
predicted_genres = mlb.inverse_transform(pipeline.predict(test_data))

# Create a DataFrame for Results
test_results = pd.DataFrame({'MOVIE_NAME': test_data['MOVIE_NAME'], 'PREDICTED_GENRES': predicted_genres})

# Replace Empty Genres with Fallback
test_results['PREDICTED_GENRES'] = test_results['PREDICTED_GENRES'].apply(lambda genres: [fallback_genre] if len(genres) == 0 else genres)

# Write Results to File
test_results.to_csv('model_predictions.txt', index=False, encoding='utf-8')

# Evaluate Model Using Validation Set
y_val_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average='micro')
recall = recall_score(y_val, y_val_pred, average='micro')
f1 = f1_score(y_val, y_val_pred, average='micro')

# Print Evaluation Metrics
print(f"Model evaluation results saved to 'model_predictions.txt'.")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print(f"Precision: {precision:.2f}\n")
print(f"Recall: {recall:.2f}\n")
print(f"F1-score: {f1:.2f}\n")