import sqlite3
import pandas as pd

def fire_data():
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

    df.to_csv('data/WA_fires.csv')



def weather_data():
    file_path = 'data/weather/'
    weather_files = ['WA_1992_1.csv', 'WA_1996_2.csv', 'WA_2001_2.csv', 'WA_2006_2.csv', 'WA_2011_2.csv',
                     'WA_1992_2.csv', 'WA_1997_1.csv', 'WA_2002_1.csv', 'WA_2007_1.csv', 'WA_2012_1.csv',
                     'WA_1993_1.csv', 'WA_1997_2.csv', 'WA_2002_2.csv', 'WA_2007_2.csv', 'WA_2012_2.csv',
                     'WA_1993_2.csv', 'WA_1998_1.csv', 'WA_2003_1.csv', 'WA_2008_1.csv', 'WA_2013_1.csv',
                     'WA_1994_1.csv', 'WA_1998_2.csv', 'WA_2003_2.csv', 'WA_2008_2.csv', 'WA_2013_2.csv',
                     'WA_1994_2.csv', 'WA_1999_1.csv', 'WA_2004_1.csv', 'WA_2009_1.csv', 'WA_2014_1.csv',
                     'WA_1994_3.csv', 'WA_1999_2.csv', 'WA_2004_2.csv', 'WA_2009_2.csv', 'WA_2014_2.csv',
                     'WA_1995_1.csv', 'WA_2000_1.csv', 'WA_2005_1.csv', 'WA_2010_1.csv', 'WA_2015_1.csv',
                     'WA_1995_2.csv', 'WA_2000_2.csv', 'WA_2005_2.csv', 'WA_2010_2.csv', 'WA_2015_2.csv',
                     'WA_1996_1.csv', 'WA_2001_1.csv', 'WA_2006_1.csv', 'WA_2011_1.csv']

    weather_features = ['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE',
                        'PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN', 'TOBS']
[value for value in variable]
    weather_data_generator = get_weather_data(file_path, weather_files,
                                              weather_features)

def get_weather_data(file_path, files, features):
    for file_name in files:
        with open(file_path + file_name) as f:
            reader = csv.reader(f)
            header = next(reader)
            col_indeces = [header.index(var) for var in features]
            for row in reader:
                yield parser(row, col_indeces)

def parser(row, col_indeces):
    return [row[i] for i in col_indeces]
