import pandas as pd
import psycopg2
import os
import sys

sys.path.append("home/ubuntu/wildfire-project/")
import predict

class DataModel():
    def __init__(self):
        conn = psycopg2.connect("host=firesdbinstance.cwspjcvdc38q.us-west-2.rds.amazonaws.com port=5432 user=Administratos password={} dbname=fires".format(os.environ['AWS_DB_PASSWORD']))
        cur = conn.cursor()

        cur.execute('''SELECT StartDate, County, Latitude, Longitude, PriorityScore
                       FROM current_fires ORDER BY PriorityScore DESC
                       LIMIT 30''')

        self.data = cur.fetchall()

        cur.close()
        conn.close()

    def get_top_fires(self):
        return self.data

    def predict_single(self, d):
        one_df = pd.DataFrame(list(d.values())).T
        one_df.columns = list(d.keys())
        one_df['date_start'] = pd.to_datetime(one_df['date_start'])
        one_df['latitude'] = pd.to_numeric(one_df['latitude'])
        one_df['longitude'] = pd.to_numeric(one_df['longitude'])
        one_df['fire_size'] = pd.to_numeric(one_df['fire_size'])
        print (one_df)
        path = '/Users/Maggie/galvanize/wildfire-project/data/temp_one_fire.csv'
        one_df.to_csv(path, index=False)
        predict.prepare_raw_data(path, path)
        predict.predict_with_score(path, path)
        one_df = pd.read_csv('../data/temp_one_fire.csv', header=None)
        one_df.columns = ['id', 'date', 'latitude', 'longitude', 'county',
                          'probability', 'area', 'return']
        one_df['probability'] = one_df['probability'].map(lambda x:"{:.2f}".format(x))
        one_df['return'] = one_df['return'].map(lambda x:"{:.0f}".format(x))
        #print(one_df.to_dict(orient='records')[0])
        return one_df.to_dict(orient='records'[0])
