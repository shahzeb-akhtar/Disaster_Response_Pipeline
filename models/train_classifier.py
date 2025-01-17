import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    '''
        tokenize text
        input: text message
        output: list of clean tokens
    '''
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
    """
        Calculate mean weighted F1 score for all of the output classifiers
        input:
            y_true: List containing actual labels.
            y_pred: List containing predicted labels.
        output:
            Mean F1 score for all of the output classifiers
    """
    f1_arr = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i], average='weighted')
        f1_arr.append(f1)
        
    score = np.mean(f1_arr)
    return score
    
def load_data(database_filepath):
    '''
        load data from a database file and split it into messages and categories values
        input: path to database file
        output: Values of messages, Values of all categories, name of all categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DataTable', con=engine)
    return df['message'].values, df.iloc[:,4:].values, list(df.columns[4:])


def build_model():
    '''
        Prepare a grid search object with parameters and pipeline of ML classifier
    '''
    #Best Parameters: {'clf__estimator__min_samples_split': 4, 'clf__estimator__n_estimators': 15, 'vect__max_df': 0.1}
    pipeline = Pipeline([
         ('vect', CountVectorizer(tokenizer=tokenize)), # tokenizer=tokenize
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(RandomForestClassifier()))
     ], verbose=True)
     
    parameters = {
        'vect__max_df': [0.10, 0.15, 0.20],         #0.10, 0.15, 0.20
        'clf__estimator__n_estimators': [10, 15, 20], # 10, 15, 20
        'clf__estimator__min_samples_split': [2, 4, 6] #2, 4, 6
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, scoring=make_scorer(performance_metric))
    return cv


def evaluate_model(model, X_test, Y_test, category_names, database_filepath):
    '''
        Evaluate the predition from model. Store evaluation metrics in a table
        input:
            model - the model to evaluate_model
            X_test - Data values on which to predict classes
            Y_test - Actual classes for X_test 
            category_names - List of all categories
            database_filepath - path of database file
        output: None
        prints the evaluation metrics for all categories as well as save it in a table in database
    '''
    y_pred = model.predict(X_test)
    result_cols = ['accuracy', 'precision', 'recall', 'f1-score']
    indices = []
    df_2d_arr = []
    for i, cat in enumerate(category_names):
        print(cat)
        print('\taccuracy: ', accuracy_score(Y_test[:,i], y_pred[:,i]), '\tprecision: ', precision_score(Y_test[:,i], y_pred[:,i], average='weighted'), '\trecall: ', recall_score(Y_test[:,i], y_pred[:,i], average='weighted'), '\tf1-score: ', f1_score(Y_test[:,i], y_pred[:,i], average='weighted'))
        print('\n')
        
        indices.append(cat)
        row = []
        row.append(accuracy_score(Y_test[:,i], y_pred[:,i]))
        row.append(precision_score(Y_test[:,i], y_pred[:,i], average='weighted'))
        row.append(recall_score(Y_test[:,i], y_pred[:,i], average='weighted'))
        row.append(f1_score(Y_test[:,i], y_pred[:,i], average='weighted'))
        df_2d_arr.append(row)
        
    result_df = pd.DataFrame(df_2d_arr, index=indices, columns=result_cols)
    
    engine = create_engine('sqlite:///' + database_filepath)
    result_df.to_sql('ResultTable', engine, index=False, if_exists= 'replace')
    

def save_model(model, model_filepath):
    '''
        Save model in a pickle file
        input:
            model - the grid search cv model to save
            model_filepath - path of pickle file
        output: None
        Saves the best estimator from the model to a pickle file
    '''
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    

def main():
    '''
        Main function - Given database file path and pickle model file path. It loads the data, builds a model. Trains and evaluates the model. Then saves it in a pickle file.
    '''
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
        evaluate_model(model, X_test, Y_test, category_names, database_filepath)

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