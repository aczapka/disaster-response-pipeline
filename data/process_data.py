import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, sep=',')
    categories = pd.read_csv(categories_filepath, sep=',')

    df = pd.merge(left=messages, right=categories, on='id', how='inner')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True).copy()

    row = categories.iloc[0, :].copy()
    category_colnames = row.values
    for i, colname in enumerate(category_colnames):
        category_colnames[i] = colname[:-2]

    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    # join messages and final categories
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    return df


def clean_data(df):
    # drop all rows with duplicated id
    df = df.drop_duplicates(subset=['id'], keep=False)

    # drop rows with categories value different from 0 or 1
    df = df.loc[~((df.drop(columns=['id', 'message', 'original', 'genre']).max(axis=1) < 0) | (
            df.drop(columns=['id', 'message', 'original', 'genre']).max(axis=1) > 1))].copy()

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_and_categories', engine, index=False)


def main():
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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
