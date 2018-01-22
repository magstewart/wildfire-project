import pandas as pd
import psycopg2

class DataModel():
    def __init__(self):
        conn = psycopg2.connect("host=firesdbinstance.cwspjcvdc38q.us-west-2.rds.amazonaws.com port=5432 user=Administratos password=IGB9837nkdywf dbname=fires")
        cur = conn.cursor()

        cur.execute('''SELECT StartDate, Latitude, Longitude, Probability
                       FROM current_fires ORDER BY Probability DESC
                       LIMIT 30''')

        self.data = cur.fetchall()

        cur.close()
        conn.close()


    def get_top_fires(self):
        return self.data
