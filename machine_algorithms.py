from data_loader import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import time
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def applyFutureSelection(X_train_vec, X_test_vec, y_train, futureSelectionAlgorithm):
    print("Applying feature selection")
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


X_train_vec, X_test_vec, y_train, y_test = getVectorizedDataFromDataFrame('spam_v2.csv', ';',0.2)


print("Train MultinomialNB algorithm")
result = testMLAlgorithm(MultinomialNB(), X_train_vec, X_test_vec, y_train, y_test)

    
# API endpoint
@app.route('/test', methods=['POST'])
@cross_origin()
def apiTestUserInput():
    try:
        data = request.get_json()
        user_input = data['user_input']

        detected_result = testUserInput(result, user_input)

        response = {
            'text': user_input,
            'prediction': detected_result
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)