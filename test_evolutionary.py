import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

data =  data = pd.read_csv('spam_NStugard.csv', sep=',', header=None, names=['label', 'text'])

df = pd.DataFrame(data)

# Text preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [ps.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

# Split the dataset into training and test sets (adjust the test_size parameter as needed)
# Split the dataset into features (X) and labels (y)
X = df['processed_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Vectorize the text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_train)
y_labels = df['label']

# Clonal Selection Algorithm (CSA)
class ArtificialImmuneSystem:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.affinity_threshold = 0.5  # Adjust based on problem characteistics
        self.population = self.initialize_population()

    def initialize_population(self):
        return [random.choices([0, 1], k=X_vectorized.shape[1]) for _ in range(self.population_size)]

    def affinity(self, antibody):
        clf = MultinomialNB()
        selected_features = [i for i, feature in enumerate(antibody) if feature == 1]
        X_selected = X_vectorized[:, selected_features]
        clf.fit(X_selected, y_train)
        predictions = clf.predict(X_selected)
        acc = accuracy_score(y_train, predictions)
        print(acc)
        return acc

    def train(self, num_iterations):
        for iteration in range(num_iterations):
            clones = []
            
            for antibody in self.population:
                clone_factor = self.affinity(antibody) / self.affinity_threshold
                num_clones = int(clone_factor * self.population_size)
                for _ in range(num_clones):
                    clones.append(self.mutate(antibody))
            
            self.population = self.select_clones(clones, self.population_size)


    def select_clones(self, antibodies, num_clones):
        sorted_antibodies = sorted(antibodies, key=lambda x: self.affinity(x), reverse=True)
        return sorted_antibodies[:num_clones]

    def mutate(self, antibody):
        return [1 - feature if random.random() < self.mutation_rate else feature for feature in antibody]

# Train the AIS
ais = ArtificialImmuneSystem(population_size=10, mutation_rate=0.1)
ais.train(num_iterations=0)

print("After trainig")

selected_antibody = ais.population[-1]
selected_feature_indices = [i for i, feature in enumerate(selected_antibody) if feature == 1]

# Vectorize the test data using only the selected features
X_test_vectorized = vectorizer.transform(X_test)
X_test_vectorized_selected = X_test_vectorized[:, selected_feature_indices]

# Train a classifier on the vectorized dataset
clf = MultinomialNB()
#Should be train vectorized
clf.fit(X_vectorized[:, selected_feature_indices], y_train)

# Make predictions on the test data
predictions = clf.predict(X_test_vectorized_selected)

# Evaluate the success rate
success_rate = accuracy_score(y_test, predictions)
print(f"Success rate on the input dataset using the last selected features: {success_rate * 100:.2f}%")

# Make predictions on the input dataset
predictions = clf.predict(X_test_vectorized_selected)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(y_test, predictions)
print("Classification Report:")
print(class_report)
