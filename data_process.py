import pandas as pd
import numpy as np
import pylab
import math

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

# def addIndicators():


def aboveBelow(num, aboveBelow, val):
    if aboveBelow == 'below':
        if num < val:
            return True
        else:
            return False


def pad_left(x, n, pad=np.nan):
    return np.pad(x, (n - x.size, 0), 'constant', constant_values=(pad,))


def add_sma(ma, column, df):
    df[f'sma{ma}'] = pad_left(
        ti.sma(np.array(df[f'{column}']), ma), df[f'{column}'].size)


def add_ema(ma, column, df):
    df[f'ema{ma}'] = pad_left(
        ti.ema(np.array(df[f'{column}']), ma), df[f'{column}'].size)


def add_vwma(df, ma):
    l = len(df.close.values)
    df[f'vwma{ma}'] = pad_left(
        ti.vwma(np.array(df['close']), np.array(df['volume']), ma), l)


def add_rsi(rsi, column, df):
    df[f'rsi{rsi}'] = pad_left(
        ti.rsi(np.array(df[f'{column}']), rsi), df[f'{column}'].size)


def add_obv(df):
    df['ovb'] = ti.obv(
        (df.close.values),
        (df.volume.values))


def add_ad(df, ad):
    df[f'ad{ad}'] = ti.ad(
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
    df[f'ultosc'] = pad_left(ti.ultosc(
        (df.high.values),
        (df.low.values),
        (df.close.values),
        short, medium, long
    ), l)


def add_vosc(df, short=2, long=5):
    l = len(df.close.values)
    df[f'vosc'] = pad_left(ti.vosc(
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
    df[f'stochk_{K}'] = pad_left(k, l)
    df[f'stochd_{K}'] = pad_left(d, l)


def add_bbands(df, period=5, stddev=2):
    l = len(df.close.values)
    (bbands_lower, bbands_middle, bbands_upper) = ti.bbands(
        (df.close.values),
        period, stddev,)
    df[f'bbands_upper'] = pad_left(bbands_upper, l)
    df[f'bbands_lower'] = pad_left(bbands_lower, l)
    df[f'bbands_middle'] = pad_left(bbands_middle, l)

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
    df['macd'] = pad_left(macd, rows)
    df['macd_signal'] = pad_left(macd_signal, rows)
    df['macd_histogram'] = pad_left(macd_histogram, rows)


def rollingApply(df, window, col, name, condFn, args=()):
    df['rolling_{window}_{name}'] = pd.Series(
        df[name]).rolling(window).apply(condFn, args=args)


def addIndicators(df):
    add_sma(2, 'close', df)
    add_sma(20, 'close', df)
    add_sma(50, 'close', df)
    add_sma(200, 'close', df)
    add_ema(2, 'close', df)
    add_ema(20, 'close', df)
    add_ema(50, 'close', df)
    add_ema(200, 'close', df)
    add_rsi(2, 'close', df)
    add_rsi(3, 'close', df)
    add_rsi(4, 'close', df)
    add_rsi(14, 'close', df)
    df.dropna()
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
    print(df.shape[0])
    for i in range(df.shape[0]):
        print(i)
        print(df.shape[0] + days_ahead)
        end = i + window_size
        if(end >= df.shape[0] + days_ahead):
            break

        window_data = df[i:end]
        target_data = df[end+days_ahead-1:end+days_ahead]
        print(window_data)
        print(target_data)
