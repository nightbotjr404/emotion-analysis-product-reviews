#Emotion Analysis for product Reviews
######Prepared by : Fahad(Group leader)(S2)
                            #Adildev(S5)
                            #Arjun(S5)
                            #Safin(S3)
                            #Alwin(S4)
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.model_selection import train_test_split

# Download NLTK stopwords if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV dataset
dataset = pd.read_csv('product_reviews2.csv')

# Check the first few rows to ensure it loaded correctly
print(dataset.head())

# Function to preprocess text
def preprocess_text(text):
    # Check if the text is not NaN (missing value)
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        # Return an empty string for missing values
        return ''

# Apply preprocessing to the text column
dataset['text'] = dataset['text'].apply(preprocess_text)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['sentiment'], test_size=0.2, random_state=42)

# Function to extract features from the text
def extract_features(text):
    words = word_tokenize(text)
    features = {}
    for word in words:
        features[word] = True
    return features

# Prepare the training data in the required NLTK format
training_data = [(extract_features(text), sentiment) for text, sentiment in zip(X_train, y_train)]

# Train the Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(training_data)

# Prepare the testing data in the required NLTK format
testing_data = [(extract_features(text), sentiment) for text, sentiment in zip(X_test, y_test)]

# Calculate accuracy
acc = accuracy(classifier, testing_data)
print("Accuracy:", acc)

# Function to analyze user input
def analyze_user_input(user_input):
    features = extract_features(user_input)
    prediction = classifier.classify(features)
    return prediction

while True:
    # Ask the user if they want to check another product review
    user_response = input("Continue ?(Y/N): ").lower()

    if user_response == 'y':
        # Ask the user for a review
        user_review = input("Enter a product review: ")

        # Get the sentiment prediction for the user's review
        predicted_sentiment = analyze_user_input(user_review)
        print("Predicted sentiment:", predicted_sentiment)

    elif user_response == 'n':
        print("Program terminated.")
        break

    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
