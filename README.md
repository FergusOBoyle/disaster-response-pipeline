# disaster-response-pipeline
Udacity Project. Disaster response message data, originally from Figure Eight, is used to build a model to classify new messages. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. In addition, some plots related to the data are displayed in the web app using Plotly.

## Packages

The analysis is delivered in the form of a Jupyter Notebook. This can be installed using Anaconda (see below).

Required packages are specified in requirements.yml.

## Files

README.md: This file.  
process_data.py: ETL pipeline. Writes output to a SQLite databases  
train_classifier.py: Trains classifier. Writes model to a pickle file  
ETL_pipeline.ipynb: sandbox jupyter notebook for developing contents of process_data.py  
ML_pipeline.ipynb: sandbox jupyter notebook for developing contents of train_classifier.py  
required_packages.yml: List of required packages for setting up of Anaconda environment.  
data/messages: Source data from udacity. Contains text messages related to disaster scenarios  
data/categories: Source data from udacity. Contains classification training data related to messages   
app/run.py: Flask file that runs app  
app/templates/go.html: classification result page of web app  
app/templates/master.html: main page of web app  

## Installation and Running

### Installing Anaconda

1. Download Anaconda from [here](https://www.anaconda.com/distribution/).
2. Install Anaconda using [these](https://docs.anaconda.com/anaconda/install/) instructions.
3. Create a new environment in the Anaconda Shell: >conda create -n my_new_env --file required_packages.yml.
4. Switch to the new environment: >conda activate my_new_env.

### Running Jupyter notebook

The jupyter notebooks can be accessed by launching jupyter notebook from the Anaconda Shell (>jupyter notebook) and navigate to the location of the disaster-response-pipeline folder.

### Running the python files

process_data.py and train_classifier.py can be run from the environment. For example:
python process_data.py 'disaster_messages.csv disaster_categories.csv

### Running the App (Instrutions are for windows only)

Execute the server: python run.py
Open the browser at: http://localhost:3001/ 

