import dataFetch as dataFetch
import data_process as data_process
import ML as ML
import pickle
import pandas as pd
from  symbols import symbols

pd.set_option('display.max_columns', None)
days_ahead = 5

def makeData(sym):
        # fetch some data
    (weekly_df, daily_df) = dataFetch.get_bars('SPY')


    # pattern
    daily_df['candle_pattern'] = data_process.add_candle_patterns(5, daily_df)    
    weekly_df['candle_pattern'] = data_process.add_candle_patterns(5, weekly_df)    

    # add some indicators
    data_process.addIndicators(weekly_df)
    data_process.addIndicators(daily_df)
    daily_df.dropna(inplace=True)
    weekly_df.dropna(inplace=True)

    weekly_df =  processData(weekly_df)
    daily_df=processData(daily_df)



    return (weekly_df, daily_df)


def processData(df):
    print(df.head())
    window_size = 20
    upper_limit = 70
    lower_limit = 30

    # add up down day count
    df['up_down_day'] = df.apply(lambda row: data_process.up_or_down(row), axis=1)
    # how many down days / up days
    data_process.add_consecutive_up_down_days(df, window_size)

    data_process.applyRollingAbove(df, 'ultosc_235', upper_limit, window=window_size)
    data_process.applyRollingBelow(df, 'ultosc_235', lower_limit, window=window_size)

    data_process.applyRollingAbove(df, 'rsi_4', upper_limit, window=window_size)
    data_process.applyRollingBelow(df, 'rsi_4', lower_limit, window=window_size)

    data_process.applyRollingAbove(df, 'rsi_8', upper_limit, window=window_size)
    data_process.applyRollingBelow(df, 'rsi_8', lower_limit, window=window_size)

    data_process.applyRollingAbove(df, 'stochrsi_5', upper_limit, window=window_size)
    data_process.applyRollingBelow(df, 'stochrsi_5', lower_limit, window=window_size)

    cci_upper_limit = 100
    cci_lower_limit = -100
    data_process.applyRollingAbove(df, 'cci_5', cci_upper_limit, window=window_size)
    data_process.applyRollingBelow(df, 'cci_5', cci_lower_limit, window=window_size)





    data_process.add_future_price_change(df, days_ahead)
    print(df.columns)

    # ML step
    rsi_df = df[[
        f'rolling_{window_size}_consecutive_up',
        f'rolling_{window_size}_consecutive_down',
        f'ultosc_235_rolling_{window_size}_below_{lower_limit}', 
        f'ultosc_235_rolling_{window_size}_above_{upper_limit}', 
        f'stochrsi_5_rolling_{window_size}_below_{lower_limit}', 
        f'stochrsi_5_rolling_{window_size}_above_{upper_limit}', 
        f'rsi_8_rolling_{window_size}_below_{lower_limit}', 
        f'rsi_8_rolling_{window_size}_above_{upper_limit}',
        f'rsi_4_rolling_{window_size}_below_{lower_limit}', 
        f'rsi_4_rolling_{window_size}_above_{upper_limit}',
        f'cci_5_rolling_{window_size}_below_{cci_lower_limit}', 
        f'cci_5_rolling_{window_size}_above_{cci_upper_limit}',
        f'target_price_change_{days_ahead}',
        # f'up_or_down'
        ]]
    print(rsi_df.head())
    print(rsi_df.tail())
    return rsi_df



def main():
# loop over symbols and make some data
    Main_daily_DF = pd.DataFrame()
    Main_weekly_DF = pd.DataFrame()
    # # READ
    # Main_daily_DF = pickle.load( open( "Main_daily_DF.p", "rb" ) )
    # Main_weekly_DF = pickle.load( open( "Main_weekly_DF.p", "rb" ) )
    for sym in symbols:
        (weekly_df, daily_df) = makeData(sym)
        Main_weekly_DF = pd.concat([Main_weekly_DF, weekly_df], ignore_index=True)
        Main_daily_DF = pd.concat([Main_daily_DF, daily_df], ignore_index=True)


    # print(weekly_df.head())
    # print(daily_df.head())
    # print(weekly_df.tail())
    # print(daily_df.tail())

#WRITE
    pickle.dump( Main_daily_DF, open( "Main_daily_DF.p", "wb" ) )
    pickle.dump( Main_weekly_DF, open( "Main_weekly_DF.p", "wb" ) )

# READ
    # Main_daily_DF = pickle.load( open( "Main_daily_DF.p", "rb" ) )
    # Main_weekly_DF = pickle.load( open( "Main_weekly_DF.p", "rb" ) )

    ML.run_ML(Main_daily_DF, f'target_price_change_{days_ahead}')
    ML.run_ML(Main_weekly_DF, f'target_price_change_{days_ahead}')

    # daily_train_data = data_process.prepairData(daily_df)
    # weekly_train_data = data_process.prepairData(weekly_df)

    # pickle these
    # pickle.dump( daily_train_data, open( "daily_train_data.p", "wb" ) )
    # pickle.dump( weekly_train_data, open( "weekly_train_data.p", "wb" ) )

# READ DATA
    # daily_train_data = pickle.load( open( "daily_train_data.p", "rb" ) )
    # weekly_train_data = pickle.load( open( "weekly_train_data.p", "rb" ) )

    # print(daily_train_data)
    # print(weekly_train_data)


main()



