from sys import argv
import src.pipeline as pipeline
import json
from src.model import Model
import pickle
import pandas as pd
import numpy as np
import sqlite3

# Run from root directory using python run.py config.json

def to_db(input_path):
    with open(config.json, 'r') as f:
        params = json.load(f)

    pipeline.test_data_pipeline(input_path,
                                params['clean_weather_data_filepath'],
                                'temporary_test_data.csv')

    X = pipeline.get_model_features('temporary_test_data.csv',
                                    params['model_features'],
                                    training_data=False)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        preds = model.predict_proba(X)[:,1]

    df = pd.read_csv('temporary_test_data.csv')
    df['prediction'] = preds
    df['fire_id'] = 123
    #df.sort_values('prediction', inplace=True, ascending=False)
    df[['date_start', 'latitude_x', 'longitude_x', 'prediction']].to_csv(
        params['predictions_filepath'], index=False)

    conn = psycopg2.connect("host=firesdbinstance.cwspjcvdc38q.us-west-2.rds.amazonaws.com port=5432 user=Administratos password=IGB9837nkdywf dbname=fires")
    cur = conn.cursor()

    

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
