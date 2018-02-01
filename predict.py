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
import boto3

# Run from root directory using python run.py config.json

def prepare_raw_data(input_path, output_path):
    config_file = '/home/ubuntu/wildfire-project/config.json'
    with open(config_file, 'r') as f:
        params = json.load(f)

    pipeline.test_data_pipeline(input_path,
                                params['clean_weather_data_filepath'],
                                output_path,
                                params['engineered_weather_features'])

def predict_with_score(input_path, output_path):
    config_file = '/home/ubuntu/wildfire-project/config.json'
    with open(config_file, 'r') as f:
        params = json.load(f)

    df = pd.read_csv(input_path)
    s3 = boto3.resource('s3')
    BUCKET_NAME = 'wildfire-project-data'

    grid_proba = pd.read_csv(s3.Bucket(BUCKET_NAME).download_file('/data/final_grid_probs.csv'), header=None)
    df['grid_prob'] = df['grid'].map(lambda x:grid_proba.iloc[int(x),1])
    X = df[params['model_features']].values
    with open('/home/ubuntu/wildfire-project/model.pkl', 'rb') as f:
        model = pickle.load(f)
        preds = model.predict_proba(X)[:,1]

    cost_per_acre = 470
    investigation_base = 300
    investigation_scale = 30
    df = pd.read_csv(input_path)
    df['prediction'] = preds
    df['expected_return'] = (df['prediction']*df['fire_size']*cost_per_acre
                             - (investigation_base +
                                investigation_scale * df['fire_size']))
    df['fire_id'] = 123
    df['expected_return'] = df['expected_return'].astype('int')
    #df.sort_values('prediction', inplace=True, ascending=False)
    df[['fire_id', 'date_start', 'latitude_x', 'longitude_x', 'fips_name',
        'prediction', 'fire_size', 'expected_return']].to_csv(
        output_path, index=False, header=False)

def write_to_db(filepath, clear_data=False):
    conn = psycopg2.connect("host=firesdbinstance.cwspjcvdc38q.us-west-2.rds.amazonaws.com port=5432 user=Administratos password={} dbname=fires".format(os.environ['AWS_DB_PASSWORD']))
    cur = conn.cursor()
    if clear_data:
        cur.execute("TRUNCATE TABLE current_fires")
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
