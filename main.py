import dataFetch as dataFetch
import data_process as data_process
import ML as ML
import pickle


def main():
    # fetch some data
    (weekly_df, daily_df) = dataFetch.get_bars('SPY')

    # add some indicators
    data_process.addIndicators(weekly_df)
    data_process.addIndicators(daily_df)
    daily_df.dropna(inplace=True)
    weekly_df.dropna(inplace=True)

    # print(weekly_df.head())
    # print(daily_df.head())
    # print(weekly_df.tail())
    # print(daily_df.tail())

    window_size = 50
    upper_limit = 60
    lower_limit = 40
    days_ahead = 5

    # data_process.applyRollingAbove(daily_df, 'rsi_2', upper_limit, window=window_size)
    # data_process.applyRollingBelow(daily_df, 'rsi_2', lower_limit, window=window_size)

    # data_process.applyRollingAbove(daily_df, 'rsi_4', upper_limit, window=window_size)
    # data_process.applyRollingBelow(daily_df, 'rsi_4', lower_limit, window=window_size)

    data_process.applyRollingAbove(daily_df, 'rsi_14', upper_limit, window=window_size)
    data_process.applyRollingBelow(daily_df, 'rsi_14', lower_limit, window=window_size)
    data_process.add_future_price_change(daily_df, days_ahead)
    print(daily_df.columns)

    # ML step
    rsi_df = daily_df[[
        # f'rsi_2_rolling_{window_size}_below_{lower_limit}', 
        # f'rsi_2_rolling_{window_size}_above_{upper_limit}', 
        f'rsi_14_rolling_{window_size}_below_{lower_limit}', 
        f'rsi_14_rolling_{window_size}_above_{upper_limit}',
        # f'rsi_4_rolling_{window_size}_below_{lower_limit}', 
        # f'rsi_4_rolling_{window_size}_above_{upper_limit}',
        f'target_price_change_{days_ahead}',
        f'up_or_down'
        ]]
    ML.run_ML(rsi_df, f'target_price_change_{days_ahead}')

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
