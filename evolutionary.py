import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

# Vectorize the text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(df['processed_text'])
y_labels = (df['label'] == 'spam').astype(int)

# Clonal Selection Algorithm (CSA)
class ArtificialImmuneSystem:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.affinity_threshold = 0.5  # Adjust based on problem characteristics
        self.population = self.initialize_population()

    def initialize_population(self):
        return [random.choices([0, 1], k=X_vectorized.shape[1]) for _ in range(self.population_size)]

    def clone(self, antibody, clone_factor):
        return [feature if random.random() > clone_factor else 1 - feature for feature in antibody]

    def mutate(self, antibody):
        return [1 - feature if random.random() < self.mutation_rate else feature for feature in antibody]

    def affinity(self, antibody):
        clf = MultinomialNB()
        clf.fit(X_vectorized[:, antibody == 1], y_labels)
        predictions = clf.predict(X_vectorized[:, antibody == 1])
        return accuracy_score(y_labels, predictions)

    def select_clones(self, antibodies, num_clones):
        sorted_antibodies = sorted(antibodies, key=lambda x: self.affinity(x), reverse=True)
        return sorted_antibodies[:num_clones]

    def train(self, num_iterations):
        for iteration in range(num_iterations):
            clones = []
            for antibody in self.population:
                clone_factor = self.affinity(antibody) / self.affinity_threshold
                num_clones = int(clone_factor * self.population_size)
                clones.extend([self.clone(antibody, clone_factor) for _ in range(num_clones)])

            self.population = clones
            self.population = self.select_clones(self.population, self.population_size)

            for i in range(self.population_size):
                self.population[i] = self.mutate(self.population[i])

# Train the AIS
ais = ArtificialImmuneSystem(population_size=10, mutation_rate=0.1)
ais.train(num_iterations=10)

# Print the best antibody selected by the AIS
best_antibody = max(ais.population, key=lambda x: ais.affinity(x))
selected_feature_indices = [i for i, feature in enumerate(best_antibody) if feature == 1]
selected_feature_names = [vectorizer.get_feature_names_out()[i] for i in selected_feature_indices]
print("Best features selected by the AIS:")
print(selected_feature_names)
