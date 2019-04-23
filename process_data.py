import sys
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Disaster message and Disaster message category data into a single dataframe

    Args:
        messages_filepath: filepath of the disaster message data (relative to ./data directory)
        categories_filepath: filepath of the disaster message category data (relative to ./data directory)
    
    Returns:
        A dataframe of the merged tables
    """
    data_directory = "./data/"
    message_file = "messages.csv"
    categories_file = "categories.csv"
    database_name = "fergus2.db"

    messages = pd.read_csv(data_directory + message_file)
    categories = pd.read_csv(data_directory + categories_file)

    df = messages.merge(categories, on='id', how='left')

    return df

def clean_data(df):
    """
    Clean the disaster response messages dataframe 

    Args:
        df: input disaster response message data
    
    Returns:
        Cleaned up disaster response message data
    """
    #first, split out the categories column into seperate columns
    cat = df['categories'].str.split(pat=';', expand=True)
    #grab the coloumn names from the first row (typical entry takes a formal similr to: "medical_help-0")
    colnames = cat.iloc[1,:].str.strip('01-')
    cat.columns = colnames
    #remove all text except the 0 or 1 for each entry
    cat = cat.replace(r'[a-z_]*-', '', regex=True)
    #can now remove the original categories column
    df = df.drop(['categories'], axis=1)
    #add these new columns to the original dataset
    complete_data = df.merge(cat, left_index=True, right_index=True)
    
    #Get rid of duplicate rows
    complete_data.drop_duplicates(subset='id', keep='first', inplace=True)

    return complete_data

def save_data(df, database_filename):
    """
    Save the data to a SQLite database file 

    Args:
        df: Cleaned input disaster response message data
        database_filename: filename to save database 
    """

    engine_name = 'sqlite:///' + database_filename
    #using sqlalchemy
    engine = create_engine(engine_name)
    df.to_sql('messages_table', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.shape)
        
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