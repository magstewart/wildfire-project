import pandas as pd
import psycopg2
import os

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
