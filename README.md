# Disaster Response Pipeline Project

### Summary of the project:

The objective of this project is to train an algorithm which will help in quickly classifying
tweets and short messages during a catastrophe. This will provide the emergency
services and the government with useful information in real time.

The dataset for truing has been downloaded from https://appen.com/

The goal is to train a NLP multi-classification algorithm on this dataset that maps each short message to
one ore more categories.

### Explanation of the files:

1. `process_data.py`: This is the ETL script which read and clean the training data and generates a database.
2. `train_classifier.py`: Training and evaluation of the model.
3. `run.py`: Runs a Flask web interface that can be used by a worker to classify new messages.

### Instructions of how to run:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

