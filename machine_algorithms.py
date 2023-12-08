from data_loader import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2


def testAlgorithm(algorithm, X_train_vec, X_test_vec, y_train, y_test):
    algorithm.fit(X_train_vec, y_train)
    predictions = algorithm.predict(X_test_vec)
    print("Class report: \n", classification_report(y_test, predictions))

def applyChiSquaredFeatureSelection(X_train_vec, X_test_vec, y_train):
    print("Applying chi squared feature selection")
    selector = SelectKBest(chi2, k=5)
    X_train_chi = selector.fit_transform(X_train_vec, y_train)
    X_test_chi = selector.transform(X_test_vec)
    return X_train_chi, X_test_chi



X_train_vec, X_test_vec, y_train, y_test = getVectorizedDataFromDataFrame('spam_NStugard.csv', ',',0.2)


X_train_vec, X_test_vec = applyChiSquaredFeatureSelection(X_train_vec, X_test_vec, y_train)

print("Testing KNeighborsClassifier algorithm")
testAlgorithm(KNeighborsClassifier(), X_train_vec, X_test_vec, y_train, y_test)

print("Testing MultinomialNB algorithm")
testAlgorithm(MultinomialNB(), X_train_vec, X_test_vec, y_train, y_test)
