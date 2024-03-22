# Import necessary libraries
import nltk
from flask import Flask, render_template, request
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
import pandas as pd

# Replace '/home/ec2-user/sentiment-analysisss/web scrping, EDA and Modeling/cleaned_data.csv' 
# with the actual path to your CSV file
df = pd.read_csv('/home/ec2-user/sentiment-analysisss/web scrping, EDA and Modeling/cleaned_data.csv')


# Preprocessing functions
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    text = text.lower()  # Convert text to lowercase
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_text)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_text)

# Apply preprocessing to the dataset
df['cleaned_review_text'] = df['cleaned_review_text'].apply(clean_text)
df['cleaned_review_text'] = df['cleaned_review_text'].apply(remove_stopwords)
df['cleaned_review_text'] = df['cleaned_review_text'].apply(lemmatize_text)

# Train the model
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_review_text'])
y = df['sentiment']
model = RandomForestClassifier()
model.fit(X, y)

# Function to predict sentiment
def predict_sentiment(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle POST request
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('index.html', sentiment=sentiment)
    else:
        # Handle GET request or initial load
        return render_template('index.html', sentiment=None)

if __name__ == '__main__':
    app.run(debug=True)
