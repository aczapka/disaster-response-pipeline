import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')


def load_data(database_filepath):
    """Loads the data from sqlite database and split it in features and labels.

    Parameters
    ----------
    database_filepath : str
        Path of the database to read.

    Returns
    -------
    X : array
        Features.
    Y : array
        Labels.
    category_names : array
        Names of the labels.

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='messages_and_categories', con=engine)
    X = df['message'].values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns.values
    return X, Y, category_names


def tokenize(text):
    """Tokenizes the data.

    Tokenization is done by forcing lower case, removing punctuation,
    tokenizing, stopowords removal and lemmatization.

    Parameters
    ----------
    text : str
        String to tokenize.

    Returns
    -------
    test : str
        Tokenized string.

    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = nltk.tokenize.word_tokenize(text)
    text = [word for word in text if not word in nltk.corpus.stopwords.words("english")]
    text = [WordNetLemmatizer().lemmatize(word) for word in text]
    return text


def build_model():
    """Creates the model.

    Returns
    -------
    cv_model
        Sklearn pipeline.

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    # grid search
    parameters = {'clf__n_estimators': [50, 100, 200]}
    cv_model = GridSearchCV(pipeline, param_grid=parameters)

    return cv_model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the modes calculation F1 score for each class.

    Parameters
    ----------
    model : sklearn pipeline
        Model to evaluate.
    X_test : array
        Test features.
    Y_test : array
        Test lables.
    category_names : array
        Label names.

    Returns
    -------
    None

    """
    # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
    Y_pred = model.predict(X_test)

    for i_class in range(np.shape(Y_test)[1]):
        print(f"Report for {category_names[i_class]} class:")
        print(classification_report(
            Y_test[:, i_class],
            Y_pred[:, i_class],
            zero_division=0
        ))


def save_model(model, model_filepath):
    """Saves the trained model to a pickle.

    Parameters
    ----------
    model : sklearn pipeline
        Model to save.
    model_filepath : string
        Path for the pickle.

    Returns
    -------
    None

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Main function."""
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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
