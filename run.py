from sys import argv
import src.make_data as generate
import src.pipeline as pipeline
import json
from src.model import Model

# Run from root directory using python run.py config.json

config_file = argv[1]

with open(config_file, 'r') as f:
    params = json.load(f)

if params['compile_fire']:
    print ('Generating fire data \n')
    generate.fire_data(params['fire_filepath'])

if params['compile_weather']:
    print ('Generating weather data\n')
    generate.weather_data(params['weather_raw_path'],
                       params['weather_raw_files'],
                       params['weather_raw_features'],
                       params['weather_filepath_out'])

if params['clean_fire_data']:
    print ('Cleaning fire data\n')
    pipeline.clean_fire(params['fire_filepath'],
                        params['clean_fire_data_filepath'])

if params['clean_weather_data']:
    print ('Cleaning weather data\n')
    pipeline.clean_weather(params['weather_filepath_out'],
                        params['clean_weather_data_filepath'],
                        params['weather_raw_features'])

if params['generate_station_coordinates']:
    print ('Generating station coordinates file\n')
    pipeline.get_station_coordinates(params['clean_weather_data_filepath'],
                                     params['stations_filepath'])

if params['add_station_to_fire']:
    print ('Matching fire data to weather stations\n')
    pipeline.add_stations_to_fire(params['clean_fire_data_filepath'],
                          params['clean_weather_data_filepath'],
                          params['fires_stations_filepath'])

if params['merge_fire_weather']:
    print ('Merging fire and weather data\n')
    pipeline.merge_fire_weather(params['fires_stations_filepath'],
                          params['clean_weather_data_filepath'],
                          params['combined_data_filepath'])

if params.get('engineer_features'):
    print ('Engineering features\n')
    pipeline.engineer_features(params['combined_data_filepath'],
                               params['engineered_data_output'])

if params.get('new_model'):
    print ('Fitting a new model\n')
    X, y = pipeline.get_model_features(params['engineered_data_output'],
                                       params['model_features'],
                                       params['model_label'],
                                       params['positive_class'])
    model = Model(params['model_type'], **params.get('model_hyperparameters'))
    if params.get('cross_validate'):
        print('Cross-validating\n')
        model.cross_validate(params.get('numnber_CV_folds'), X, y,
                             params.get('staged_predict'))
