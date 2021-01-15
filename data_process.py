import pandas as pd
import numpy as np
import pylab
import math
import talib
from talib import abstract

from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
# from statsmodels.stats.outliers_infulence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sklearn
import tulipy as ti


# print(talib.get_functions())
# for groups in talib.get_function_groups():
#     print(groups)
#     for func in talib.get_function_groups()[groups]:
#         print(func)
#         print(abstract.Function(func))

def add_candle_patterns(days, df):
    vals = [0 for i in range(days)]
    for n in range(len(df)-days):
        val = pattern_recognition(df[n:n+days])
        vals.append(val)
    return vals


def pattern_recognition(df):
    val = 0
    # test_len = 5
    dates = np.array(df.index)
    open = np.array(df.open)
    close = np.array(df.close)
    high = np.array(df.high)
    low = np.array(df.low)
    # volume = np.array(df.volume)
    tristar = talib.CDLTRISTAR(open, high, low, close)
    Two_Crows = talib.CDL2CROWS(open, high, low, close)
    Three_Black_Crows = talib.CDL3BLACKCROWS(open, high, low, close)
    Three_Inside_Up_Down = talib.CDL3INSIDE(open, high, low, close)
    Three_Line_Strike = talib.CDL3LINESTRIKE(open, high, low, close)
    Three_Outside_Up_Down = talib.CDL3OUTSIDE(open, high, low, close)
    Three_Stars_In_The_South = talib.CDL3STARSINSOUTH(open, high, low, close)
    Three_Advancing_White_Soldiers = talib.CDL3WHITESOLDIERS(
        open, high, low, close)
    Abandoned_Baby = talib.CDLABANDONEDBABY(
        open, high, low, close, penetration=0)
    Advance_Block = talib.CDLADVANCEBLOCK(open, high, low, close)
    Belt_hold = talib.CDLBELTHOLD(open, high, low, close)
    Breakaway = talib.CDLBREAKAWAY(open, high, low, close)
    Closing_Marubozu = talib.CDLCLOSINGMARUBOZU(open, high, low, close)
    Concealing_Baby_Swallow = talib.CDLCONCEALBABYSWALL(open, high, low, close)
    Counterattack = talib.CDLCOUNTERATTACK(open, high, low, close)
    Dark_Cloud_Cover = talib.CDLDARKCLOUDCOVER(
        open, high, low, close, penetration=0)
    Doji = talib.CDLDOJI(open, high, low, close)
    Doji_Star = talib.CDLDOJISTAR(open, high, low, close)
    Dragonfly_Doji = talib.CDLDRAGONFLYDOJI(open, high, low, close)
    Engulfing_Pattern = talib.CDLENGULFING(open, high, low, close)
    Evening_Doji_Star = talib.CDLEVENINGDOJISTAR(
        open, high, low, close, penetration=0)
    Evening_Star = talib.CDLEVENINGSTAR(open, high, low, close, penetration=0)
    Up_Down_gap_side_by_side_white_lines = talib.CDLGAPSIDESIDEWHITE(
        open, high, low, close)
    Gravestone_Doji = talib.CDLGRAVESTONEDOJI(open, high, low, close)
    Hammer = talib.CDLHAMMER(open, high, low, close)
    Hanging_Man = talib.CDLHANGINGMAN(open, high, low, close)
    Harami_Pattern = talib.CDLHARAMI(open, high, low, close)
    Harami_Cross_Pattern = talib.CDLHARAMICROSS(open, high, low, close)
    High_Wave_Candle = talib.CDLHIGHWAVE(open, high, low, close)
    Hikkake_Pattern = talib.CDLHIKKAKE(open, high, low, close)
    Modified_Hikkake_Pattern = talib.CDLHIKKAKEMOD(open, high, low, close)
    Homing_Pigeon = talib.CDLHOMINGPIGEON(open, high, low, close)
    Identical_Three_Crows = talib.CDLIDENTICAL3CROWS(open, high, low, close)
    In_Neck_Pattern = talib.CDLINNECK(open, high, low, close)
    Inverted_Hammer = talib.CDLINVERTEDHAMMER(open, high, low, close)
    Kicking = talib.CDLKICKING(open, high, low, close)
    Kicking_bull_bear_determined_by_the_longer_marubozu = talib.CDLKICKINGBYLENGTH(
        open, high, low, close)
    Ladder_Bottom = talib.CDLLADDERBOTTOM(open, high, low, close)
    Long_Legged_Doji = talib.CDLLONGLEGGEDDOJI(open, high, low, close)
    Long_Line_Candle = talib.CDLLONGLINE(open, high, low, close)
    Marubozu = talib.CDLMARUBOZU(open, high, low, close)
    Matching_Low = talib.CDLMATCHINGLOW(open, high, low, close)
    Mat_Hold = talib.CDLMATHOLD(open, high, low, close, penetration=0)
    Morning_Doji_Star = talib.CDLMORNINGDOJISTAR(
        open, high, low, close, penetration=0)
    Morning_Star = talib.CDLMORNINGSTAR(open, high, low, close, penetration=0)
    On_Neck_Pattern = talib.CDLONNECK(open, high, low, close)
    Piercing_Pattern = talib.CDLPIERCING(open, high, low, close)
    Rickshaw_Man = talib.CDLRICKSHAWMAN(open, high, low, close)
    Rising_Falling_Three_Methods = talib.CDLRISEFALL3METHODS(
        open, high, low, close)
    Separating_Lines = talib.CDLSEPARATINGLINES(open, high, low, close)
    Shooting_Star = talib.CDLSHOOTINGSTAR(open, high, low, close)
    Short_Line_Candle = talib.CDLSHORTLINE(open, high, low, close)
    Spinning_Top = talib.CDLSPINNINGTOP(open, high, low, close)
    Stalled_Pattern = talib.CDLSTALLEDPATTERN(open, high, low, close)
    Stick_Sandwich = talib.CDLSTICKSANDWICH(open, high, low, close)
    Takuri_Dragonfly_Doji_with_very_long_lower_shadow = talib.CDLTAKURI(
        open, high, low, close)
    Tasuki_Gap = talib.CDLTASUKIGAP(open, high, low, close)
    Thrusting_Pattern = talib.CDLTHRUSTING(open, high, low, close)
    Tristar_Pattern = talib.CDLTRISTAR(open, high, low, close)
    Unique_3_River = talib.CDLUNIQUE3RIVER(open, high, low, close)
    Upside_Gap_Two_Crows = talib.CDLUPSIDEGAP2CROWS(open, high, low, close)
    Upside_Downside_Gap_Three_Methods = talib.CDLXSIDEGAP3METHODS(
        open, high, low, close)
    pattern_arrays = [tristar, Two_Crows, Three_Black_Crows, Three_Inside_Up_Down, Three_Line_Strike, Three_Outside_Up_Down, Three_Stars_In_The_South, Three_Advancing_White_Soldiers, Abandoned_Baby, Advance_Block, Belt_hold, Breakaway, Closing_Marubozu, Concealing_Baby_Swallow, Counterattack, Dark_Cloud_Cover, Doji, Doji_Star, Dragonfly_Doji, Engulfing_Pattern, Evening_Doji_Star, Evening_Star, Up_Down_gap_side_by_side_white_lines, Gravestone_Doji, Hammer, Hanging_Man, Harami_Pattern, Harami_Cross_Pattern, High_Wave_Candle, Hikkake_Pattern, Modified_Hikkake_Pattern, Homing_Pigeon,
                      Identical_Three_Crows, In_Neck_Pattern, Inverted_Hammer, Kicking, Kicking_bull_bear_determined_by_the_longer_marubozu, Ladder_Bottom, Long_Legged_Doji, Long_Line_Candle, Marubozu, Matching_Low, Mat_Hold, Morning_Doji_Star, Morning_Star, On_Neck_Pattern, Piercing_Pattern, Rickshaw_Man, Rising_Falling_Three_Methods, Separating_Lines, Shooting_Star, Short_Line_Candle, Spinning_Top, Stalled_Pattern, Stick_Sandwich, Takuri_Dragonfly_Doji_with_very_long_lower_shadow, Tasuki_Gap, Thrusting_Pattern, Tristar_Pattern, Unique_3_River, Upside_Gap_Two_Crows, Upside_Downside_Gap_Three_Methods]

    pattern_names = ['tristar', 'Two_Crows', 'Three_Black_Crows', 'Three_Inside_Up_Down', 'Three_Line_Strike', 'Three_Outside_Up_Down', 'Three_Stars_In_The_South', 'Three_Advancing_White_Soldiers', 'Abandoned_Baby', 'Advance_Block', 'Belt_hold', 'Breakaway', 'Closing_Marubozu', 'Concealing_Baby_Swallow', 'Counterattack', 'Dark_Cloud_Cover', 'Doji', 'Doji_Star', 'Dragonfly_Doji', 'Engulfing_Pattern', 'Evening_Doji_Star', 'Evening_Star', 'Up_Down_gap_side_by_side_white_lines', 'Gravestone_Doji', 'Hammer', 'Hanging_Man', 'Harami_Pattern', 'Harami_Cross_Pattern', 'High_Wave_Candle', 'Hikkake_Pattern', 'Modified_Hikkake_Pattern', 'Homing_Pigeon',
                     'Identical_Three_Crows', 'In_Neck_Pattern', 'Inverted_Hammer', 'Kicking', 'Kicking_bull_bear_determined_by_the_longer_marubozu', 'Ladder_Bottom', 'Long_Legged_Doji', 'Long_Line_Candle', 'Marubozu', 'Matching_Low', 'Mat_Hold', 'Morning_Doji_Star', 'Morning_Star', 'On_Neck_Pattern', 'Piercing_Pattern', 'Rickshaw_Man', 'Rising_Falling_Three_Methods', 'Separating_Lines', 'Shooting_Star', 'Short_Line_Candle', 'Spinning_Top', 'Stalled_Pattern', 'Stick_Sandwich', 'Takuri_Dragonfly_Doji_with_very_long_lower_shadow', 'Tasuki_Gap', 'Thrusting_Pattern', 'Tristar_Pattern', 'Unique_3_River', 'Upside_Gap_Two_Crows', 'Upside_Downside_Gap_Three_Methods']

    for pattern_results in range(len(pattern_arrays)):
        # print(f'Found {pattern_names[pattern_results]}')
        # print(pattern_arrays[pattern_results])
        for index in range(len(pattern_arrays[pattern_results])):
            value = pattern_arrays[pattern_results][index]
            val = val + value
            # if value != 0:
    #             print(
    #                 f'{pattern_arrays[pattern_results]} index - {index} value - {value}')
    # print('-----------------------------------------------')
    # print(f'Total val = {val}')
    return val


def up_or_down(row, target):
    if(row[target] > 0):
        return 1
    else:
        return 0


def add_future_price_change(df, days):
    df[f'target_price_change_{days}'] = (
        (df['close'].shift(-days) - df['close'])/df['close'])*100
    # df[f'up_or_down'] = df.apply(lambda row: up_or_down(
    #     row, f'target_price_change_{days}'), axis=1)


def countAbove(val, _list):
    count = 0
    for n in range(len(list(_list))):
        # print(n)
        # print(list(_list))
        # print(list(_list)[n])
        if list(_list)[n] > val:
            count = count+1
    return count


def countBelow(val, _list):
    count = 0
    for n in range(len(list(_list))):
        # print(n)
        # print(list(_list))
        # print(list(_list)[n])
        if list(_list)[n] < val:
            count = count+1
    return count


def applyRollingAbove(df, name, limit, window):
    df[f'{name}_rolling_{window}_above_{limit}'] = df[f'{name}'].rolling(
        window).apply(lambda x: countAbove(limit, x))


def applyRollingBelow(df, name, limit, window):
    df[f'{name}_rolling_{window}_below_{limit}'] = df[f'{name}'].rolling(
        window).apply(lambda x: countBelow(limit, x))


def runIsBetween(low, high, data):
    count = 0
    for i in range(0, len(data)):
        data_point = float(data[i:i+1])
        if data_point < high and data_point > low:
            count = count+1
    return count


def runAboveBelow(aboveOrBelow, value, data):
    count = 0
    for i in range(0, len(data)):
        data_point = float(data[i:i+1])
        isAboveOrBelow = aboveBelow(data_point, aboveOrBelow, value)
        if isAboveOrBelow:
            count = count+1
    return count


def aboveBelow(num, aboveOrBelow, val):
    val = float(val)
    num = float(num)
    if aboveOrBelow == 'below':
        if num < val:
            return True
        else:
            return False

    if aboveOrBelow == 'above':
        if num > val:
            return True
        else:
            return False


def pad_left(x, n, pad=np.nan):
    return np.pad(x, (n - x.size, 0), 'constant', constant_values=(pad,))


def add_sma(ma, column, df):
    df[f'sma_{ma}'] = pad_left(
        ti.sma(np.array(df[f'{column}']), ma), df[f'{column}'].size)


def add_ema(ma, column, df):
    df[f'ema_{ma}'] = pad_left(
        ti.ema(np.array(df[f'{column}']), ma), df[f'{column}'].size)


def add_vwma(df, ma):
    l = len(df.close.values)
    df[f'vwma_{ma}'] = pad_left(
        ti.vwma(np.array(df['close']), np.array(df['volume']), ma), l)


def add_rsi(rsi, column, df):
    df[f'rsi_{rsi}'] = pad_left(
        ti.rsi(np.array(df[f'{column}']), rsi), df[f'{column}'].size)


def add_obv(df):
    df['ovb'] = ti.obv(
        (df.close.values),
        (df.volume.values))


def add_ad(df, ad):
    df[f'ad_{ad}'] = ti.ad(
        (df.high.values),
        (df.low.values),
        (df.close.values),
        (df.volume.values))


def add_willr(df, period):
    # df[f'willr_{period}'] =
    df[f'willr_{period}'] = pad_left(ti.willr(
        (df.high.values),
        (df.low.values),
        (df.close.values),
        period
    ), len(df.high.values))


def add_roc(df, period):
    df[f'roc_{period}'] = pad_left(ti.roc(
        np.array(df.close), period), len(np.array(df.close)))

    df[f'rocr_{period}'] = pad_left(ti.rocr(
        np.array(df.close), period), len(np.array(df.close)))


def add_cmo(df, period):
    # the Chande Momentum Oscillator
    df[f'cmo_{period}'] = pad_left(ti.cmo(
        np.array(df.close), period), len(np.array(df.close)))


def add_mom(df, period):
    # the Chande Momentum Oscillator
    df[f'mom_{period}'] = pad_left(ti.mom(
        np.array(df.close), period), len(np.array(df.close)))


def add_stochrsi(df, period):
    df[f'stochrsi_{period}'] = pad_left(ti.stochrsi(
        np.array(df.close), period), len(np.array(df.close)))


def add_bop(df):
    l = len(df.close.values)
    df['bop'] = pad_left(ti.bop(
        (df.open.values),
        (df.high.values),
        (df.low.values),
        (df.close.values),
    ), l)


def add_mfi(df, period=5):
    l = len(df.close.values)
    df[f'mfi_{period}'] = pad_left(ti.mfi(
        (df.high.values),
        (df.low.values),
        (df.close.values),
        (df.volume.values), period
    ), l)


def add_ultosc(df, short=2, medium=3, long=5):
    l = len(df.close.values)
    df[f'ultosc_{short}{medium}{long}'] = pad_left(ti.ultosc(
        (df.high.values),
        (df.low.values),
        (df.close.values),
        short, medium, long
    ), l)


def add_vosc(df, short=2, long=5):
    l = len(df.close.values)
    df[f'vosc_{short}{long}'] = pad_left(ti.vosc(
        (df.volume.values),
        short,  long
    ), l)


def add_stoch(df, K=5, period=3, D=3):
    l = len(df.close.values)
    (k, d) = ti.stoch(
        (df.high.values),
        (df.low.values),
        (df.close.values),
        K, period, D,)
    df[f'stochk_{K}{period}{D}'] = pad_left(k, l)
    df[f'stochd_{K}{period}{D}'] = pad_left(d, l)


def add_bbands(df, period=5, stddev=2):
    l = len(df.close.values)
    (bbands_lower, bbands_middle, bbands_upper) = ti.bbands(
        (df.close.values),
        period, stddev,)
    df[f'bbands_upper_{period}{stddev}'] = pad_left(bbands_upper, l)
    df[f'bbands_lower_{period}{stddev}'] = pad_left(bbands_lower, l)
    df[f'bbands_middle_{period}{stddev}'] = pad_left(bbands_middle, l)

# indicates trends as bullish or bearish


def add_adx(df, period=10):
    l = len(df.close.values)
    df[f'adx_{period}'] = pad_left(ti.adx(
        np.array(df.high),
        np.array(df.low),
        np.array(df.close), period), l)


def add_cci(df, period=5):
    l = len(df.close.values)
    df[f'cci_{period}'] = pad_left(ti.cci(
        np.array(df.high),
        np.array(df.low),
        np.array(df.close), period), l)


def add_atr(df, period=5):
    l = len(df.close.values)
    df[f'atr_{period}'] = pad_left(ti.atr(
        np.array(df.high),
        np.array(df.low),
        np.array(df.close), period), l)


# shift in concensus bull vs bear
def add_macd(df, short=6, long=15, signal=20):
    # opt = {'short period':6, 'long period':15, 'signal period':20}
    rows = df.shape[0]
    (macd, macd_signal, macd_histogram) = ti.macd(
        np.array(df.close), short, long, signal)
    df[f'macd_{short}{long}{signal}'] = pad_left(macd, rows)
    df[f'macd_signal_{short}{long}{signal}'] = pad_left(macd_signal, rows)
    df[f'macd_histogram_{short}{long}{signal}'] = pad_left(
        macd_histogram, rows)


# def rollingApply(df, window, col, name, condFn, args=()):
#     df['rolling_{window}_{name}'] = pd.Series(
#         df[name]).rolling(window).apply(condFn, args=args)


def addIndicators(df):

    inputs = {
        'open': np.array(df.open),
        'high': np.array(df.close),
        'low': np.array(df.high),
        'close': np.array(df.low),
        'volume': np.array(df.volume)
    }
    print(talib.abstract.Function('STOCH'))
    # slowk, slowd = talib.STOCH(inputs, 5, 3, 0, 3, 0)

    # add_sma(2, 'close', df)
    # add_sma(20, 'close', df)
    # add_sma(50, 'close', df)
    # add_sma(200, 'close', df)
    add_ema(2, 'close', df)
    add_ema(20, 'close', df)
    add_ema(50, 'close', df)
    add_ema(200, 'close', df)
    add_rsi(2, 'close', df)
    add_rsi(3, 'close', df)
    add_rsi(4, 'close', df)
    add_rsi(8, 'close', df)
    add_rsi(14, 'close', df)

    add_ad(df, 5)
    add_macd(df)
    add_adx(df)
    add_cmo(df, 5)
    add_roc(df, 10)
    add_willr(df, 10)
    add_stochrsi(df, 5)
    add_stoch(df, 5, 3, 3)
    add_obv(df)
    add_mom(df, 5)
    add_bop(df)
    add_bbands(df, 5, 2)
    add_cci(df, 5)
    add_atr(df, 5)
    add_mfi(df, 5)
    add_ultosc(df, 2, 3, 5)
    add_vosc(df, 2, 5)
    add_vwma(df, 5)


def prepairData(df):
    # in the last 20, 30, 50 bars, how often was the RSI above/below 50? 70? below 30
    window_size = 50
    days_ahead = 5
    indicator = 'rsi_14'
    high_indicator_value = 60
    low_indicator_value = 40
    print(df.shape[0])
    # training_df = pd.DataFrame( columns=['low_rsi_count', 'middle_rsi_count', 'high_rsi_count', 'perc_change'])
    training_df = []
    for i in range(df.shape[0]):
        print(i)
        print(df.shape[0])
        end = i + window_size
        print(end+days_ahead-1)
        if(end+days_ahead-1 >= df.shape[0]):
            break
        # continue
        window_data = df[indicator][i:end]
        target_price = df['close'][end+days_ahead-1:end+days_ahead]
        current_price = df['close'][end-1]
        current_time = df.index[end-1]
        # print(window_data)
        print(target_price)
        # print(window_data.columns)
        """
        ['open', 'high', 'low', 'close', 'volume', 'sma_2', 'sma_20', 'sma_50',
       'sma_200', 'ema_2', 'ema_20', 'ema_50', 'ema_200', 'rsi_2', 'rsi_3',
       'rsi_4', 'rsi_14', 'ad_5', 'macd_61520', 'macd_signal_61520',
       'macd_histogram_61520', 'adx_10', 'cmo_5', 'roc_10', 'rocr_10',
       'willr_10', 'stochrsi_5', 'stochk_533', 'stochd_533', 'ovb', 'mom_5',
       'bop', 'bbands_upper_52', 'bbands_lower_52', 'bbands_middle_52',
       'cci_5', 'atr_5', 'mfi_5', 'ultosc_235', 'vosc_25', 'vwma_5']
       """
        #    what percentage of time was rsi above 70
        high_rsi_count = runAboveBelow(
            'above', high_indicator_value, window_data)
        low_rsi_count = runAboveBelow(
            'below', low_indicator_value, window_data)
        middle_rsi_count = runIsBetween(
            low_indicator_value, high_indicator_value, window_data)
        target_price = float(target_price)
        # print(window_data.index[-1])
        current_time = window_data.index[-1]
        perc_change = abs(current_price - target_price) / current_price
        training_df.append([current_time, low_rsi_count,
                            middle_rsi_count, high_rsi_count, perc_change])
        # print(high_rsi_count, low_rsi_count, middle_rsi_count, perc_change)
        new_df = pd.DataFrame([row], columns=[
                              'low_rsi_count', 'middle_rsi_count', 'high_rsi_count', 'perc_change'])
        training_df.append(new_df, ignore_index=True)
        print(training_df.head())
        print(training_df.tail())
        print(training_df)
    training_df = pd.DataFrame(training_df, columns=[
                               'datetime', 'low_rsi_count', 'middle_rsi_count', 'high_rsi_count', 'perc_change'])
    training_df.index = training_df['datetime']
    training_df.drop('datetime', axis=1, inplace=True)

    print(training_df.head())
    print(training_df.tail())

    return training_df


def add_consecutive_up_down_days(df, window):
    _window = window
    if window > 5: _window = 5
    df[f'rolling_{window}_consecutive_up'] = df['up_down_day'].rolling(
        _window).apply(lambda x: consecutiveUp(x))

    df[f'rolling_{window}_consecutive_down'] = df['up_down_day'].rolling(
        _window).apply(lambda x: consecutiveDown(x))


def consecutiveUp(days):
    count = 0
    for n in days:

        if n > 0:
            count = count+1
        else:
            count = 0
    return count


def consecutiveDown(days):
    count = 0
    for n in days:

        if n < 0:
            count = count+1
        else:
            count = 0
    return count


def countBelow(val, _list):
    count = 0
    for n in range(len(list(_list))):
        # print(n)
        # print(list(_list))
        # print(list(_list)[n])
        if list(_list)[n] < val:
            count = count+1
    return count


def up_or_down(row):
    if row['open'] > row['close']:
        return 1
    if row['open'] < row['close']:
        return -1
    return 0


"""
def pattern_recognition(df):
    test_len = 5
    dates = np.array(df.index[0:test_len])
    open = np.array(df.open[0:test_len])
    close = np.array(df.close[0:test_len])
    high = np.array(df.high[0:test_len])
    low = np.array(df.low[0:test_len])
    volume = np.array(df.volume[0:test_len])
    tristar = talib.CDLTRISTAR(open, high, low, close)
    Two_Crows = talib.CDL2CROWS(open, high, low, close)
    Three_Black_Crows = talib.CDL3BLACKCROWS(open, high, low, close)
    Three_Inside_Up_Down = talib.CDL3INSIDE(open, high, low, close)
    Three_Line_Strike = talib.CDL3LINESTRIKE(open, high, low, close)
    Three_Outside_Up_Down = talib.CDL3OUTSIDE(open, high, low, close)
    Three_Stars_In_The_South = talib.CDL3STARSINSOUTH(open, high, low, close)
    Three_Advancing_White_Soldiers = talib.CDL3WHITESOLDIERS(
        open, high, low, close)
    Abandoned_Baby = talib.CDLABANDONEDBABY(
        open, high, low, close, penetration=0)
    Advance_Block = talib.CDLADVANCEBLOCK(open, high, low, close)
    Belt_hold = talib.CDLBELTHOLD(open, high, low, close)
    Breakaway = talib.CDLBREAKAWAY(open, high, low, close)
    Closing_Marubozu = talib.CDLCLOSINGMARUBOZU(open, high, low, close)
    Concealing_Baby_Swallow = talib.CDLCONCEALBABYSWALL(open, high, low, close)
    Counterattack = talib.CDLCOUNTERATTACK(open, high, low, close)
    Dark_Cloud_Cover = talib.CDLDARKCLOUDCOVER(
        open, high, low, close, penetration=0)
    Doji = talib.CDLDOJI(open, high, low, close)
    Doji_Star = talib.CDLDOJISTAR(open, high, low, close)
    Dragonfly_Doji = talib.CDLDRAGONFLYDOJI(open, high, low, close)
    Engulfing_Pattern = talib.CDLENGULFING(open, high, low, close)
    Evening_Doji_Star = talib.CDLEVENINGDOJISTAR(
        open, high, low, close, penetration=0)
    Evening_Star = talib.CDLEVENINGSTAR(open, high, low, close, penetration=0)
    Up_Down_gap_side_by_side_white_lines = talib.CDLGAPSIDESIDEWHITE(
        open, high, low, close)
    Gravestone_Doji = talib.CDLGRAVESTONEDOJI(open, high, low, close)
    Hammer = talib.CDLHAMMER(open, high, low, close)
    Hanging_Man = talib.CDLHANGINGMAN(open, high, low, close)
    Harami_Pattern = talib.CDLHARAMI(open, high, low, close)
    Harami_Cross_Pattern = talib.CDLHARAMICROSS(open, high, low, close)
    High_Wave_Candle = talib.CDLHIGHWAVE(open, high, low, close)
    Hikkake_Pattern = talib.CDLHIKKAKE(open, high, low, close)
    Modified_Hikkake_Pattern = talib.CDLHIKKAKEMOD(open, high, low, close)
    Homing_Pigeon = talib.CDLHOMINGPIGEON(open, high, low, close)
    Identical_Three_Crows = talib.CDLIDENTICAL3CROWS(open, high, low, close)
    In_Neck_Pattern = talib.CDLINNECK(open, high, low, close)
    Inverted_Hammer = talib.CDLINVERTEDHAMMER(open, high, low, close)
    Kicking = talib.CDLKICKING(open, high, low, close)
    Kicking_bull_bear_determined_by_the_longer_marubozu = talib.CDLKICKINGBYLENGTH(
        open, high, low, close)
    Ladder_Bottom = talib.CDLLADDERBOTTOM(open, high, low, close)
    Long_Legged_Doji = talib.CDLLONGLEGGEDDOJI(open, high, low, close)
    Long_Line_Candle = talib.CDLLONGLINE(open, high, low, close)
    Marubozu = talib.CDLMARUBOZU(open, high, low, close)
    Matching_Low = talib.CDLMATCHINGLOW(open, high, low, close)
    Mat_Hold = talib.CDLMATHOLD(open, high, low, close, penetration=0)
    Morning_Doji_Star = talib.CDLMORNINGDOJISTAR(
        open, high, low, close, penetration=0)
    Morning_Star = talib.CDLMORNINGSTAR(open, high, low, close, penetration=0)
    On_Neck_Pattern = talib.CDLONNECK(open, high, low, close)
    Piercing_Pattern = talib.CDLPIERCING(open, high, low, close)
    Rickshaw_Man = talib.CDLRICKSHAWMAN(open, high, low, close)
    Rising_Falling_Three_Methods = talib.CDLRISEFALL3METHODS(
        open, high, low, close)
    Separating_Lines = talib.CDLSEPARATINGLINES(open, high, low, close)
    Shooting_Star = talib.CDLSHOOTINGSTAR(open, high, low, close)
    Short_Line_Candle = talib.CDLSHORTLINE(open, high, low, close)
    Spinning_Top = talib.CDLSPINNINGTOP(open, high, low, close)
    Stalled_Pattern = talib.CDLSTALLEDPATTERN(open, high, low, close)
    Stick_Sandwich = talib.CDLSTICKSANDWICH(open, high, low, close)
    Takuri_Dragonfly_Doji_with_very_long_lower_shadow = talib.CDLTAKURI(
        open, high, low, close)
    Tasuki_Gap = talib.CDLTASUKIGAP(open, high, low, close)
    Thrusting_Pattern = talib.CDLTHRUSTING(open, high, low, close)
    Tristar_Pattern = talib.CDLTRISTAR(open, high, low, close)
    Unique_3_River = talib.CDLUNIQUE3RIVER(open, high, low, close)
    Upside_Gap_Two_Crows = talib.CDLUPSIDEGAP2CROWS(open, high, low, close)
    Upside_Downside_Gap_Three_Methods = talib.CDLXSIDEGAP3METHODS(
        open, high, low, close)
    pattern_arrays = [tristar, Two_Crows, Three_Black_Crows, Three_Inside_Up_Down, Three_Line_Strike, Three_Outside_Up_Down, Three_Stars_In_The_South, Three_Advancing_White_Soldiers, Abandoned_Baby, Advance_Block, Belt_hold, Breakaway, Closing_Marubozu, Concealing_Baby_Swallow, Counterattack, Dark_Cloud_Cover, Doji, Doji_Star, Dragonfly_Doji, Engulfing_Pattern, Evening_Doji_Star, Evening_Star, Up_Down_gap_side_by_side_white_lines, Gravestone_Doji, Hammer, Hanging_Man, Harami_Pattern, Harami_Cross_Pattern, High_Wave_Candle, Hikkake_Pattern, Modified_Hikkake_Pattern, Homing_Pigeon,
                      Identical_Three_Crows, In_Neck_Pattern, Inverted_Hammer, Kicking, Kicking_bull_bear_determined_by_the_longer_marubozu, Ladder_Bottom, Long_Legged_Doji, Long_Line_Candle, Marubozu, Matching_Low, Mat_Hold, Morning_Doji_Star, Morning_Star, On_Neck_Pattern, Piercing_Pattern, Rickshaw_Man, Rising_Falling_Three_Methods, Separating_Lines, Shooting_Star, Short_Line_Candle, Spinning_Top, Stalled_Pattern, Stick_Sandwich, Takuri_Dragonfly_Doji_with_very_long_lower_shadow, Tasuki_Gap, Thrusting_Pattern, Tristar_Pattern, Unique_3_River, Upside_Gap_Two_Crows, Upside_Downside_Gap_Three_Methods]

    pattern_names = ['tristar', 'Two_Crows', 'Three_Black_Crows', 'Three_Inside_Up_Down', 'Three_Line_Strike', 'Three_Outside_Up_Down', 'Three_Stars_In_The_South', 'Three_Advancing_White_Soldiers', 'Abandoned_Baby', 'Advance_Block', 'Belt_hold', 'Breakaway', 'Closing_Marubozu', 'Concealing_Baby_Swallow', 'Counterattack', 'Dark_Cloud_Cover', 'Doji', 'Doji_Star', 'Dragonfly_Doji', 'Engulfing_Pattern', 'Evening_Doji_Star', 'Evening_Star', 'Up_Down_gap_side_by_side_white_lines', 'Gravestone_Doji', 'Hammer', 'Hanging_Man', 'Harami_Pattern', 'Harami_Cross_Pattern', 'High_Wave_Candle', 'Hikkake_Pattern', 'Modified_Hikkake_Pattern', 'Homing_Pigeon',
                      'Identical_Three_Crows', 'In_Neck_Pattern', 'Inverted_Hammer', 'Kicking', 'Kicking_bull_bear_determined_by_the_longer_marubozu', 'Ladder_Bottom', 'Long_Legged_Doji', 'Long_Line_Candle', 'Marubozu', 'Matching_Low', 'Mat_Hold', 'Morning_Doji_Star', 'Morning_Star', 'On_Neck_Pattern', 'Piercing_Pattern', 'Rickshaw_Man', 'Rising_Falling_Three_Methods', 'Separating_Lines', 'Shooting_Star', 'Short_Line_Candle', 'Spinning_Top', 'Stalled_Pattern', 'Stick_Sandwich', 'Takuri_Dragonfly_Doji_with_very_long_lower_shadow', 'Tasuki_Gap', 'Thrusting_Pattern', 'Tristar_Pattern', 'Unique_3_River', 'Upside_Gap_Two_Crows', 'Upside_Downside_Gap_Three_Methods']
    [print(i, d) for i, d in enumerate(dates)]
    for pattern_results in range(len(pattern_arrays)):
        print(f'Found {pattern_names[pattern_results]}')
        for index in range(len(pattern_arrays[pattern_results])):
            value = pattern_arrays[pattern_results][index] 
            if value != 0:
                print(
                    f'{pattern_arrays[pattern_results]} index - {index} value - {value}')

"""
