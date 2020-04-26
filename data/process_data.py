import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        load data from a messages and categories CSV files
        input:
            messages_filepath - path to messages csv file
            categories_filepath - path to categories csv file
        output:
            Data frame including both messages and categories together
    '''
    #read csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    merged_df = messages.merge(categories, on=['id'])
    
    return merged_df
    
    
def clean_data(df):
    '''
        Clean the data
        input:
            df - Data Frame with raw data
        output:
            Cleaned data frame
    '''
     # create a dataframe of the 36 individual category columns
    categories_split = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories_split dataframe
    row = categories_split.iloc[0]
    
    # use this row to extract a list of new column names for categories
    category_colnames = []
    for category in row:
        category_colnames.append(category.split('-')[0])

    # rename the columns of `categories_split` df
    categories_split.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories_split:
        # set each value to be the last character of the string and convert to numeric
        categories_split[column] = pd.to_numeric(categories_split[column].str[-1], downcast='integer')
    
    #drop any column with all zero - we cant predict on all 0s
    categories_split = categories_split[categories_split.columns[categories_split.sum()>0]]
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    new_df = df.join(categories_split, on='id', how='inner')

    # drop duplicates
    new_df.drop_duplicates(inplace=True)
    
    # expect only 0 and 1
    if 'related' in new_df.columns:
        indicesToDrop = new_df[new_df['related'] == 2].index
        if len(indicesToDrop) > 0:
            new_df.drop(indicesToDrop , inplace=True)
    
    return new_df

def save_data(df, database_filename):
    '''
        Savs data in a SQLite database
        input:
            df - Data frame to save
            database_filename - path of database file
        output: None
            Saves the dataframe in a tabel in database file
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DataTable', engine, index=False, if_exists= 'replace')  


def main():
    '''
        Main function.
        Given the path of  messages and categories CSV files and path to database file. It loads the CSV data, cleans it and saves it in the database file.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()