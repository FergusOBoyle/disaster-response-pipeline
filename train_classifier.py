import sys
import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Load ML data from database and return features, X and targets, Y 

    Args:
        database_filepath (str): filepath of database
    
    Returns:
        X (dataframe): feature variables
        Y (dataframe): target variables
    """
    engine_name = 'sqlite:///' + database_filepath
    engine = create_engine(engine_name)
    df =pd.read_sql("SELECT * FROM messages_table", engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    return X, Y 


def tokenize(text):
    """
    returns a tokenized, lemmatized and normalized version of text

    Args:
        text (str): input text to be tokenized, lemmatized and normalized 
    
    Returns:
        Tokenized, lemmatized and normalized version of text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model(X_train, Y_train):
    """
    Create pipline model, fit and optimize using randomized gridsearch 

    Args:
        X_train (dataframe): feature variables to be used for training 
        Y_train (dataframe): target variables to be used for training

    Returns:
        Optimized model 
    """
    #Choosing a straighforward single tree model to make training tractable in terms of time
    DTC = DecisionTreeClassifier(random_state = 11)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=DTC))
    ])

    parameters = {'clf__estimator__criterion': ["gini", "entropy"],
                  'clf__estimator__splitter': ["best", "random"],
                  'clf__estimator__max_depth': randint(3, 6),
                  'clf__estimator__min_samples_split': randint(2,6)}

    grid_obj = RandomizedSearchCV(pipeline,parameters,n_iter=5, cv=5 )
    grid_obj.fit(X_train, Y_train)

    return grid_obj.best_estimator_

def print_score(y_actual, y_pred, measure):
    """
    Creates a pretty print of the results of sklearns classification report comparing y_actual and y_pred

    Args:
        y_actual (dataframe): expected values
        y_pred (dataframe): predicted values
        measure (str): choice of measure ('weighted avg','micro avg','macro avg' )
    """
    print("\t\tWeighted Average Scores Over Each Output Class\n")
    print("\t\tPrecision\tRecall\t\tF1_Score")
    for column_name, column in y_actual.iteritems():
        report  = classification_report(y_actual[column_name], y_pred[column_name], output_dict=True )
        prec = report[measure]['precision']
        recall =  report[measure]['recall']
        f1 = report[measure]['f1-score']
        print("%20.2f %15.2f % 15.2f" % (prec, recall, f1) + "\t\t" + column_name )

def evaluate_model(model, X_test, Y_test):
    """
    Runs test data through a model and creates an evaluation report of the results

    Args:
        model (model object): model to be used for evaulation
        y_actual (dataframe): expected values
        y_pred (dataframe): predicted values
    """   
    #Make predictions with the model
    Y_pred = model.predict(X_test)
    #convert numpy output to dataframe and add columns
    Y_pred_df = pd.DataFrame(Y_pred)
    Y_pred_df.columns = Y_test.columns
    #Convert predictions and correct y values to float for faciliate comparison
    Y_pred_df = Y_pred_df.astype('float64')
    Y_test = Y_test.astype('float64')
    print_score(Y_test, Y_pred_df, 'weighted avg')


def save_model(model, model_filepath):
    """
    Saves pickle file of the model

    Args:
        model (model object): model to be saved in pickle file
        model_filepath: filepath to save to
    """ 
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, percentage_data = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        #Lets us select smaller sample sizes to faciliate quicker debug
        len_full_dataset = len(X_train)
        sample_size = int((len_full_dataset/100)*int(percentage_data))

        print('Building model...')
        model = build_model(X_train[:sample_size], Y_train[:sample_size])
         
        print('Training model...')
        model.fit(X_train[:sample_size], Y_train[:sample_size])
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument, the filepath of the pickle file to '\
              'save the model to as the second argument and a percentage of'\
              'the input data to be used (from 1 to 100). \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl 50')


if __name__ == '__main__':
    main()