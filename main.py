import dataFetch as dataFetch
import data_process as data_process

def main():
    # fetch some data
    (weekly_df, daily_df) = dataFetch.get_bars('SPY')

    # add some indicators
    data_process.addIndicators(weekly_df)
    data_process.addIndicators(daily_df)
    daily_df.dropna(inplace=True)
    weekly_df.dropna(inplace=True)
    print(weekly_df.head())
    print(daily_df.head())
    print(weekly_df.tail())
    print(daily_df.tail())
    daily = data_process.prepairData(daily_df)
    weekly = data_process.prepairData(weekly_df)
main()
