import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd


data = pd.read_csv('spam_NStugard.csv', sep=',', header=None, names=['label', 'text'])


df = pd.DataFrame(data)

# Text preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [ps.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=5)


#print(X_test)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Feature selection using chi-squared
#k_best = 5  # You can adjust this parameter based on your needs
#selector = SelectKBest(chi2, k=k_best)
#X_train_selected = selector.fit_transform(X_train_vectorized, y_train)
#X_test_selected = selector.transform(X_test_vectorized)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Make predictions
predictions = clf.predict(X_test_vectorized)


# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Method for user input
def is_spam(input_text):
    processed_text = preprocess_text(input_text)
    input_vectorized = vectorizer.transform([processed_text])
    prediction = clf.predict(input_vectorized)[0]
    return prediction

# Test the user nput method
user_want_to_test = False
while(user_want_to_test):
    user_input = input("Enter a text to test if it's spam or not: ")
    result = is_spam(user_input)
    print(f"The input text is predicted as: {result}")
