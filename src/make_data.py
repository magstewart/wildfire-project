import sqlite3
import pandas as pd
import csv

def fire_data(output_file):
    '''
    Pulls Washington data from wildfire data base.  Writes the desired
    columns to a csv file.

    Input:
    ------
    output_file: String specifying the path where the data will be written.

    Output:
    -------
    None
    '''

    conn = sqlite3.connect("data/FPA_FOD_20170508.sqlite")
    cur = conn.cursor()

    df = pd.read_sql_query('''SELECT FIRE_YEAR,
                                     DISCOVERY_DATE,
                                     DISCOVERY_DOY,
                                     DISCOVERY_TIME,
                                     STAT_CAUSE_CODE,
                                     STAT_CAUSE_DESCR,
                                     CONT_DATE,
                                     CONT_DOY,
                                     CONT_TIME,
                                     FIRE_SIZE,
                                     FIRE_SIZE_CLASS,
                                     LATITUDE,
                                     LONGITUDE,
                                     STATE,
                                     COUNTY,
                                     FIPS_CODE,
                                     FIPS_NAME
                                     FROM Fires
                                     WHERE STATE = 'WA';''', conn)

    cur.close()
    conn.close()

    df.to_csv('data/WA_fires.csv', index=False)


def weather_data(file_path, weather_files, weather_features, output_file):
    '''
    Pulls desired columns from raw weather data and compiles all into one csv

    Input:
    ------
    file_path: String specifying path for the raw weather data directory
    weather_files: List of strings specifying weather file names
    weather_features: List of strings specifying columns from raw weather
                      data to include in output
    output_file: String specifying the path where the data will be written.

    Output:
    -------
    None
    '''
    weather_data_generator = get_weather_data(file_path, weather_files,
                                              weather_features)
    with open('output_file', 'a') as f:
        fwriter = csv.writer(f, delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        for row in weather_data_generator:
            fwriter.writerow(row)


def get_weather_data(file_path, files, features):
    '''
    Returns a generator that iterates over rows of weather data with the
    desired columns

    Input:
    ------
    file_path: String specifying path for the raw data directory
    files: List of strings specifying file names
    features: List of strings specifying columns from raw data to include in output
    output_file: String specifying the path where the data will be written.

    Output:
    -------
    None
    '''
    for file_name in files:
        with open(file_path + file_name) as f:
            reader = csv.reader(f)
            header = next(reader)
            col_indeces = [header.index(var) for var in features]
            for row in reader:
                yield parser(row, col_indeces)

def parser(row, col_indeces):
    '''
    Returns row of data with the desired columns

    Input:
    ------
    row: A list containing onw row of data
    col_indeces: A list containing the indeces for the desired columns

    Output:
    -------
    A list of data
    '''
    return [row[i] for i in col_indeces]
