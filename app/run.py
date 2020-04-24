import json
import plotly
import pandas as pd
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine, inspect


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DataTable', engine)
results_df = pd.read_sql_table('ResultTable', engine)

#prepare viz data
categories_df = df.iloc[:,4:]

#data for bar chart
sum_df = pd.DataFrame(categories_df.sum(), columns=['Value'])
sum_df['Name'] = sum_df.index

#data for co-relation between categories matrix

corr_mat = categories_df.corr()
#insert indices as first columns
corr_mat.insert(0, 'Name', corr_mat.index)
#insert columns as first row
new_row = pd.Series(corr_mat.columns)
row_df = pd.DataFrame(new_row).T
row_df.columns = corr_mat.columns
corr_mat = pd.concat([row_df, corr_mat], ignore_index=True)
# convert to 2d array
corr_arr = corr_mat.to_numpy()
#make sure [0,0] element is blank as required by JS library
corr_arr[0,0] = ''

#same steps as for corr df
results_df.insert(0, 'Name', categories_df.columns)
new_row = pd.Series(results_df.columns)
row_df = pd.DataFrame(new_row).T
row_df.columns = results_df.columns
results_df = pd.concat([row_df, results_df], ignore_index=True)
results_arr = results_df.to_numpy()
results_arr[0,0] = ''

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # render web page send data for charts
    return render_template('master.html', catsJSON=sum_df.to_json(orient='records'), corr_arr=corr_arr.tolist(), results_arr = results_arr.tolist())

# web page that handles user query and displays model results
@app.route('/predict', methods=['POST'])
def predict():
    # save user input in query
    query = request.form.getlist('query')[0]    
    
    predicted_classes = []
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    
    for i, col in enumerate(list(df.columns[4:])):
        if(classification_labels[i] == 1):
            predicted_classes.append(col)
    
    return jsonify({'result': predicted_classes})
    #return render_template('go.html', predictions=predicted_classes)
    

def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()