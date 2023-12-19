from data_loader import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import time

def applyFutureSelection(X_train_vec, X_test_vec, y_train, futureSelectionAlgorithm):
    print("Applying chi squared feature selection")
    selector = SelectKBest(futureSelectionAlgorithm, k=8)
    X_train = selector.fit_transform(X_train_vec, y_train)
    X_test = selector.transform(X_test_vec)
    return X_train, X_test

def testMLAlgorithm(algorithm, X_train_vec, X_test_vec, y_train, y_test, futureSelectionAlgorithm=None):
    if(futureSelectionAlgorithm != None):
        X_train_vec, X_test_vec = applyFutureSelection(X_train_vec, X_test_vec, y_train, futureSelectionAlgorithm)
    algorithm.fit(X_train_vec, y_train)
    predictions = algorithm.predict(X_test_vec)
    print("Class report: \n", classification_report(y_test, predictions, digits=4))
    print("Confusion matrix: \n", confusion_matrix(y_test, predictions))
    return algorithm
    
def testUserInput(algorithm, text):
    vectorized_text = getVectorizedTextWithLemmantization(text)
    result = algorithm.predict(vectorized_text)
    return f"Input text is {result[0]}" 


#X_train_vec, X_test_vec, y_train, y_test = getVectorizedDataFromDataFrame('spam_NStugard.csv', ',',0.2, TfidfVectorizer())
X_train_vec, X_test_vec, y_train, y_test = getVectorizedDataFromDataFrame('spam_v2.csv', ';',0.2)
print("Testing results with TFID Vectorizer")

print("Testing RandomForestClassifier algorithm")
#testMLAlgorithm(RandomForestClassifier(), X_train_vec, X_test_vec, y_train, y_test, mutual_info_classif)

print("Testing MultinomialNB algorithm")
#testMLAlgorithm(MultinomialNB(), X_train_vec, X_test_vec, y_train, y_test, mutual_info_classif)

print("Testing support vector algorithm")
#testMLAlgorithm(SVC(), X_train_vec, X_test_vec, y_train, y_test, mutual_info_classif)




#X_train_vec, X_test_vec, y_train, y_test = getVectorizedDataFromDataFrame('spam_NStugard.csv', ',',0.2, CountVectorizer())
X_train_vec, X_test_vec, y_train, y_test = getVectorizedDataFromDataFrame('spam_v2.csv', ';',0.2)
print("##########################################################################################")
print("##########################################################################################")
print("##########################################################################################")
print("Testing with CountVectorizer")
print("Testing RandomForestClassifier algorithm")
#testMLAlgorithm(KNeighborsClassifier(), X_train_vec, X_test_vec, y_train, y_test, chi2)

print("Testing MultinomialNB algorithm")
result = testMLAlgorithm(MultinomialNB(), X_train_vec, X_test_vec, y_train, y_test)

print("Testing support vector algorithm")
#testMLAlgorithm(SVC(), X_train_vec, X_test_vec, y_train, y_test, chi2)

while(True):
    user_input = input("Text to test: ")
    tic = time.perf_counter()
    print(testUserInput(result, user_input))
    toc = time.perf_counter()
    print(f"Message was classified in {toc - tic:0.8f} seconds")