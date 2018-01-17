import pandas as pd
import numpy as np
import pyspark as ps

def clean_fire(input_path, output_path):
    '''
    Converts DISCOVERY_DATE from julian data to datetime and creates other
    date related columns. Converts column names to lower case.

    Input:
    ------
    input_path: String specifying path for the data to be cleaned
    output_path: String specifying the path where the cleaned dataframe will
    be saved

    Output:
    -------
    None
    '''
    # import all fire data to pandas
    df = pd.read_csv(input_path)
    #df = df.drop('Unnamed: 0', axis=1)

    # Column names
    columns = df.columns
    columns = columns.str.lower()
    df.columns = columns

    # convert DISCOVERY_DATE from julian date to datetime and add month column
    epoch = pd.to_datetime(0, unit='s').to_julian_date()
    df['date_start']=pd.to_datetime(df['discovery_date'] - epoch, unit='D')
    df['month'] = df['date_start'].map(lambda x : x.month)
    df['date_end']=pd.to_datetime(df['cont_date'] - epoch, unit='D')
    df['length'] = df['date_end'] - df['date_start']
    df['length'] = df['length'].map(lambda x : x.days)


    df.to_csv(output_path, index=False)


def clean_weather(input_path, output_path, weather_raw_features):
    '''
    Converts the column names to lower case.  Fills in missing values and
    creates date related columns.

    Input:
    ------
    input_path: String specifying path for the data to be cleaned
    output_path: String specifying the path where the cleaned dataframe will
    be saved

    Output:
    -------
    None
    '''
    df = pd.read_csv(input_path, header=None)
    df.columns = [col.lower() for col in weather_raw_features]

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].map(lambda x : x.month)
    df['year'] = df['date'].map(lambda x : x.year)
    df['doy'] = df['date'].map(lambda x : x.dayofyear)

    # Fill missing values
    df_sorted = df.sort_values(['year', 'doy', 'longitude', 'latitude'])
    df_sorted['snow'].fillna(value=0.0, inplace=True)
    df_sorted['snwd'].fillna(value=0.0, inplace=True)
    df_sorted['tmax'].fillna(method='ffill', inplace=True)
    df_sorted['tmin'].fillna(method='ffill', inplace=True)
    df_sorted['prcp'].fillna(method='ffill', inplace=True)

    df_sorted.to_csv(output_path, index=False)


def get_station_coordinates(input_path, output_path):
    '''
    Creates a csv file containing all the weather stations and their
    latitude and longitude.

    Input:
    ------
    input_path: String specifying path for the weather data.
    output_path: String specifying the path where the stations coordiantes
    will be saved.

    Output:
    -------
    None
    '''
    weather = pd.read_csv(input_path)
    group_station = weather.groupby('station')
    station_coordinates = group_station[['latitude', 'longitude']].max()
    station_coordinates.to_csv(output_path)


def add_stations_to_fire(fire_filepath, weather_filepath, output_file):
    '''
    Joins fire data with weather data from the station closest to each fire.

    Input:
    ------
    fire_filepath: String specifying path for the fire data.
    weather_filepath: String specifying path for the weather data.
    output_path: String specifying the path where the joined data
    will be saved.

    Output:
    -------
    None
    '''
    fires = pd.read_csv(fire_filepath)
    weather = pd.read_csv(weather_filepath)
    stations = weather[['station', 'latitude', 'longitude', 'doy', 'year']]

    v_get_station = np.vectorize(get_nearby_station, excluded=['stations'])
    fires['weather_station'] = v_get_station(lat=fires['latitude'],
                                             lon=fires['longitude'],
                                             year=fires['fire_year'],
                                             doy=fires['discovery_doy'],
                                             stations=stations)

    fires.to_csv(output_file, index=False)


def get_nearby_station(lat, lon, year, doy, stations):
    '''
    Returns the name of the station closest to the input latitude and longitude
    with weather records on that date

    Input:
    ------
    lat: float representing the latitude of the fire location
    lon: float representing the longitude of the fire location
    year: int representing the fire year
    doy: int representing the day of year
    stations: dataframe of weather data containing latitude, longitude, year
    and day of year

    Output:
    -------
    string containing the station id
    '''
    stations_date = stations[(stations['year'] == year) &
                           (stations['doy'] == doy)]
    stations_date['distance'] = np.sqrt((stations_date.loc[:,'latitude']-lat)**2 +
                                    (stations_date.loc[:,'longitude']-lon)**2)
    if len(stations_date['distance'].values) > 0:
        closest_index = np.argmin(stations_date['distance'].values)
        return stations_date.iloc[closest_index,0]
    else:
        return None

def merge_fire_weather(fire_filepath, weather_filepath, output_path):
    fires = pd.read_csv(fire_filepath)
    weather = pd.read_csv(weather_filepath)
    combined = pd.merge(fires, weather, how='left',
                    left_on=['weather_station', 'fire_year', 'discovery_doy'],
                    right_on=['station', 'year', 'doy'])
    combined.to_csv(output_path, index=False)


if __name__ == '__main__':
    spark = (ps.sql.SparkSession.builder
            .master("local[4]")
            .appName("weather")
            .getOrCreate())

    df_weather = spark.read.csv('data/all_weather.csv',
                         header=True,       # use headers or not
                         quote='"',         # char for quotes
                         sep=",",           # char for separation
                         inferSchema=True)

    clean_weather_spark(df_weather)

    #clean_fire()
    #clean_weather()
    #get_station_coordinates()
    #combine_data()
