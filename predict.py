from sys import argv
import src.pipeline as pipeline
import json
from src.model import Model
import pickle
import pandas as pd
import numpy as np
import sqlite3
import psycopg2
import csv
import os

# Run from root directory using python run.py config.json

def predict_on_raw_data(input_path, output_path):
    config_file = '/Users/Maggie/galvanize/wildfire-project/config.json'
    with open(config_file, 'r') as f:
        params = json.load(f)

    pipeline.test_data_pipeline(input_path,
                                params['clean_weather_data_filepath'],
                                output_path)

    X = pipeline.get_model_features(output_path,
                                    params['model_features'],
                                    training_data=False)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        preds = model.predict_proba(X)[:,1]

    df = pd.read_csv(output_path)
    df['prediction'] = preds
    df['fire_id'] = 123
    #df.sort_values('prediction', inplace=True, ascending=False)
    df[['fire_id', 'date_start', 'latitude_x', 'longitude_x', 'county',
        'prediction', 'fire_size', 'prediction']].to_csv(
        output_path, index=False, header=False)

def write_to_db(filepath):
    conn = psycopg2.connect("host=firesdbinstance.cwspjcvdc38q.us-west-2.rds.amazonaws.com port=5432 user=Administratos password={} dbname=fires".format(os.environ['AWS_DB_PASSWORD']))
    cur = conn.cursor()
    with open(filepath) as f:
        cur.copy_from(f, 'current_fires', sep=',')

    conn.commit()
    cur.close()
    conn.close()



if __name__ == '__main__':
    config_file = argv[1]

    with open(config_file, 'r') as f:
        params = json.load(f)


    if params.get('prepare_test_data'):
        print('Preparing test data\n')
        pipeline.test_data_pipeline(params['raw_test_data_path'],
                                    params['clean_weather_data_filepath'],
                                    params['clean_test_data_path'])

    if params.get('make_predictions'):
        print('Predicting with model\n')
        X = pipeline.get_model_features(params['clean_test_data_path'],
                                        params['model_features'],
                                        training_data=False)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            preds = model.predict_proba(X)[:,1]

        df = pd.read_csv(params['clean_test_data_path'])
        df['prediction'] = preds
        df.sort_values('prediction', inplace=True, ascending=False)
        df[['date_start', 'latitude_x', 'longitude_x', 'prediction']].to_csv(
            params['predictions_filepath'], index=False)
