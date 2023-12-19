from pandas import read_csv, DataFrame
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
nltk.download('wordnet')

vectorizer = CountVectorizer()

def getDataFrame(file_name, separator):
    data = read_csv(file_name, sep=separator, header=None, names=['label', 'text'])
    return DataFrame(data)
    
def preprocessOnlyRemovingStopWords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if not word.lower() in stop_words]
    return ' '.join(words)

def preprocessWithStemming(text):
    stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()
    words = word_tokenize(text)
    words = [porter_stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

def preprocessWithLemmantization(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(words)


def splitAndVectorizeData(data_frame, test_size):
    processed_text = data_frame['text'].apply(preprocessWithLemmantization)
    X_train, X_test, y_train, y_test = train_test_split(processed_text, data_frame['label'], test_size=test_size, random_state=5)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, y_train, y_test

def getVectorizedTextWithLemmantization(text):
    lemmatized = preprocessWithLemmantization(text)
    return vectorizer.transform([lemmatized])

def getVectorizedDataFromDataFrame(file_name, separator, test_size):
    data_frame = getDataFrame(file_name, separator)
    return splitAndVectorizeData(data_frame, test_size)

