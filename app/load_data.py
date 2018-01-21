import pandas as pd

class DataModel():
    def __init__(self):
        self.data = pd.read_csv("~/galvanize/wildfire-project/data/predictions.csv")

    def get_top_fires(self):
        return self.data.head(5).to_dict("records")
