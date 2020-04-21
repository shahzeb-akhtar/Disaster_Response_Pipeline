import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, f1_score, precision_score, recall_score, accuracy_score

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
# Define performance metric for use in grid search scoring object
def performance_metric(y_true, y_pred):
    """Calculate median F1 score for all of the output classifiers
        Args:
        y_true: array. Array containing actual labels.
        y_pred: array. Array containing predicted labels.
        Returns:
        score: float. Mean F1 score for all of the output classifiers
        """
    f1_arr = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i], average='weighted')
        f1_arr.append(f1)
        
    score = np.mean(f1_arr)
    return score
    
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, con=engine)
    return df['message'].values, df.iloc[:,4:].values, list(df.columns[4:])


def build_model():
    #Best Parameters: {'clf__estimator__min_samples_split': 4, 'clf__estimator__n_estimators': 15, 'vect__max_df': 0.1}
    pipeline = Pipeline([
         ('vect', CountVectorizer(tokenizer=tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(RandomForestClassifier()))
     ], verbose=True)
     
    parameters = {
        'vect__max_df': [0.10, 0.15, 0.20],
        'clf__estimator__n_estimators': [10, 15, 20],       
        'clf__estimator__min_samples_split': [2, 4, 6]      
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, scoring=make_scorer(performance_metric))
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    for i, cat in enumerate(category_names):
        print(cat)
        print('\taccuracy: ', accuracy_score(Y_test[:,i], y_pred[:,i]), '\tprecision: ', precision_score(Y_test[:,i], y_pred[:,i], average='weighted'), '\trecall: ', recall_score(Y_test[:,i], y_pred[:,i], average='weighted'), '\tf1-score: ', f1_score(Y_test[:,i], y_pred[:,i], average='weighted'))
        print('\n')


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()