from pymongo import MongoClient
import pandas as pd


client = MongoClient()
db=client.stock_app


def get_bars(symbol):
    bars = db.stockdatas.find({'symbol':symbol})
    bars = list(bars)[0]
    # bars_df = pd.DataFrame(bars)

    weekly_df = set_df(bars['weekly'])
    daily_df = set_df(bars['daily'])

    return (weekly_df, daily_df)

def set_df(data):
    df = pd.DataFrame(data)
    df = df.astype({'volume':'float64'})
    df['datetime']= pd.to_datetime(df['datetime'], unit = 'ms', errors='coerce')
    df.index = df.datetime
    df.drop('datetime', axis=1, inplace=True)

    
    return df
