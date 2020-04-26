# Disaster Response Pipeline Project
Classify messages, received during a disaster, into appropriate categories so relevant teams can help.

## Table of Contents
 <ol>
   <li><a href="#head1"> Libraries and tools used</a>
   <li><a href="#head2"> Motivation of the project </a>
   <li><a href="#head3"> Directory structure </a>
   <li><a href="#head4"> Summary of the results </a>
   <li><a href="#head5"> Acknowledgements </a>
   <li><a href="#head6"> Author </a>
</ol>

<h2 id="head1"> Libraries and tools used: </h2>
<ul>
 <li> NumPy
 <li> pandas
 <li> NLTK
 <li> pickle
 <li> SQLAlchemy 
 <li> scikit-learn
 <li> Flask    
 <li> d3
</ul>

<h2 id="head2"> Motivation of the project</h2>

To apply NLP skills learned as part of Udacity Data Science Nanodegree lesson to classify messages received during a disaster. It will allow relevant teams to get the message quickly and act accordingly.  

<h2 id="head3"> Directory structure </h3>

```
.
├── Datasets                                               # Datasets used as the source of analysis 
    ├── 2011_Stack_Overflow_Survey_Results.csv
    ├── 2012_Stack_Overflow_Survey_Results.csv
    ├── 2013_Stack_Overflow_Survey_Responses.csv
    ├── 2014_Stack_Overflow_Survey_Responses.csv
    ├── 2015_Stack_Overflow_Developer_Survey_Responses.csv
    ├── 2016_Stack_Overflow_Survey_Responses.csv
    ├── 2017_survey_results_public.csv
    ├── 2018_survey_results_public.csv
    ├── 2019_survey_results_public.csv 
├── Datasets                                                # Images used 
    ├── Top Languages.png
    ├── Languages_Comparison.png
    ├── Databases_Comparison.png
    ├── Platforms_Comparison.png
    ├── Frameworks_Comparison.png    
├── Stack_Overflow_Survey_Analysis.ipynb                    # Jupyter notebook for main analysis
├── Stackoverflow_2019_Survey_Exploration.ipynb             # Jupyter notebook forexploration of 2019 survey
├── README.md                                               # ReadMe file

```

<h2 id="head4"> Summary of the results </h2>
Below are few charts summarizing the results. These charts as well a working model can be accessed at https://disaster-webapp-viz.herokuapp.com/

Chart belows the distribution of categories in the training data

Chart below shows the co-relation between categories

Chart below shows the performance of the model across categories

<h2 id="head5"> Acknowledgements </h2>

<ul>
 <li> Udacity https://www.udacity.com/
 <li> Figure Eight https://appen.com/datasets/combined-disaster-response-data/
</ul>

<h2 id="head6"> Author </h2>

Shahzeb Akhtar

https://www.linkedin.com/in/shahzebakhtar/


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
