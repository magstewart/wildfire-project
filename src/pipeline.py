import pandas as pd
import numpy as np
import boto3

def clean_fire(input_path, output_path, counties=False):
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
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Column names
    columns = df.columns
    columns = columns.str.lower()
    df.columns = columns

    # convert DISCOVERY_DATE from julian date to datetime and add month column
    if 'discovery_date' in df.columns:
        epoch = pd.to_datetime(0, unit='s').to_julian_date()
        df['date_start']=pd.to_datetime(df['discovery_date'] - epoch, unit='D')
    else:
        df['date_start'] = pd.to_datetime(df['date_start'])
        df['fire_year'] = df['date_start'].map(lambda x : x.year)
        df['discovery_doy'] = df['date_start'].map(lambda x : x.dayofyear)
    df['month'] = df['date_start'].map(lambda x : x.month)
    if 'cont_date' in df.columns:
        df['date_end']=pd.to_datetime(df['cont_date'] - epoch, unit='D')
        df['length'] = df['date_end'] - df['date_start']
        df['length'] = df['length'].map(lambda x : x.days)

    df.drop_duplicates(inplace=True)

    if counties:
        geolocator = Nominatim()
        df['fips_name'] = np.vectorize(get_county, excluded=['geolocator'])(
                           df['latitude'], df['longitude'], geolocator)

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

def get_county(lat, lon, geolocator):
    location = geolocator.reverse("{}, {}".format(lat, lon))
    return location.raw['address']['county'][:-7]

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
    '''
    Performs a left joing on the fire data and weather data based on the nearest
    station label on the fire data, the year and day of year.

    Input:
    ------
    fire_filepath: string containing the filepath for the fire data
    weather_filepath: string containing the filepath for the weather data
    output_path: string containing the filepath where the merged data will
                 be saved

    Output:
    -------
    None
    '''
    fires = pd.read_csv(fire_filepath)

    weather = pd.read_csv(weather_filepath)
    combined = pd.merge(fires, weather, how='left',
                    left_on=['weather_station', 'fire_year', 'discovery_doy'],
                    right_on=['station', 'year', 'doy'])
    combined['tmax'].loc[combined['tmax'] > 150] = combined['tmax'].loc[combined['tmax'] > 150]/100
    combined = combined.drop(['latitude_y', 'longitude_y', 'station', 'date',
                              'month_y', 'doy', 'year'], axis=1)
    combined.drop_duplicates(inplace=True)
    combined.to_csv(output_path, index=False)

def engineer_features(input_filepath, weather_filepath,
                      output_filepath, features, training_data=True):
    '''
    Adds engineeref feature columns to the data set

    Input:
    ------
    input_filepath: string containing the filepath for input data
    output_path: string containing the filepath where new data will be saved

    Output:
    -------
    None
    '''
    df = pd.read_csv(input_filepath)

    weather = pd.read_csv(weather_filepath)
    df['date_start'] = pd.to_datetime(df['date_start'])
    weather['date'] = pd.to_datetime(weather['date'])
    for feature in features:
        print ('engineering {}'.format(feature['name']))
        df_merge = concat_weather_feature(df, feature['window'], feature['col'],
                                          feature['metric'], weather)
        df_merge.columns=['station', 'date', feature['name']]
        df = df.merge(df_merge, how='left', left_on=['weather_station', 'date_start'],
                      right_on=['station', 'date'], copy=False)
        df.drop(['station', 'date'], axis=1, inplace=True)

    df = get_grid(df)

    if training_data:
        df['cause_group'] = np.vectorize(group_cause)(df['stat_cause_descr'])

    df.to_csv(output_filepath, index=False)

def weather_feature(window, col, station, metric, weather):
    temp_df = weather.loc[weather['station']==station][['station','date', col]]
    temp_df = temp_df.sort_values('date')
    if metric == 'sum':
        return temp_df.rolling(window, on='date').sum()
    if metric == 'mean':
        return temp_df.rolling(window, on='date').mean()

def concat_weather_feature(df, window, col, metric, weather):
    df_temp = pd.DataFrame()
    for station in df['weather_station'].unique():
        df_temp = pd.concat([df_temp, weather_feature(window, col,
                            station, metric, weather)])
    return df_temp

def get_grid(df):
    df['lat_bin'] = (df.latitude_x - 45.56) // 0.35
    df['long_bin'] = (df.longitude_x + 124.72) // 0.39
    df['long_bin'].loc[df['long_bin']>19] = 19
    df['long_bin'].loc[df['long_bin']<0] = 0
    df['lat_bin'].loc[df['lat_bin']>9] = 9
    df['lat_bin'].loc[df['lat_bin']<0] = 0
    df['grid'] = df['lat_bin']*20+df['long_bin']
    return df

def split_final_test(input_path, train_path, test_path):
    df = pd.read_csv(input_path)
    df[df['fire_year'] == 2015].to_csv(test_path, index=False)
    df[(df['fire_year'] < 2015) & (df['fire_year'] > 1991)].to_csv(train_path, index=False)



def get_model_features(filepath, features, label=None, positive_class=None,
                        training_data=True):
    '''
    Returns numpy arrays ready for use in an sklearn model

    Input:
    ------
    filepath: string containing the filepath for input data
    features: list of strings representing the features from the input data
              that will be used in the model
    label: string representing the label column name to be returned as y
    positive_class: name of the positive class to be converted to 1s

    Output:
    -------
    X: numpy array of features
    y: numpy array of labels
    '''
    df = pd.read_csv(filepath)
    X = df[features].values
    if training_data:
        df[label] = np.vectorize(group_cause)(df['stat_cause_descr'])
        y = df[label].values
        y = y == positive_class
        return X, y
    return X

def group_cause(cause):
    '''
    Returns returns 'human' when cause is in the list of human causes, otherwise
    returns 'other'

    Input:
    ------
    cause: string containing the cause label

    Output:
    -------
    The cause group: 'human' or 'other'
    '''
    human_activity = ['Debris Burning', 'Campfire', 'Arson', 'Children', 'Fireworks', 'Smoking', 'Equipment Use']
    other = ['Missing/Undefined', 'Powerline', 'Railroad', 'Structure', 'Lightning', 'Miscellaneous']
    #nature = ['Lightning']

    if cause in human_activity:
        return 'human'
    elif cause in other:
        return 'other'

def test_data_pipeline(input_path, weather_filepath, output_path, features):
    clean_fire(input_path, output_path, False)

    add_stations_to_fire(output_path, weather_filepath, output_path)
    merge_fire_weather(output_path, weather_filepath, output_path)

    engineer_features(output_path, weather_filepath, output_path, features, False)


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
