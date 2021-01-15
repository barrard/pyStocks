import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans


def run_ML(_df, target):
    print('Start of some ML?')
    # print(_df.head())
    # print(_df.tail())
    print(_df.columns)
    df = _df.dropna()
    # print(df)
    # print(df.describe())
    print(df.head())
    print(df.tail())

    y = df[target]
    X = df.drop(target, axis=1)
    print(f'Data Len = {len(df)}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)
    print(y_test)
    # (knn_predictions, knn_test) = k_nearest_neighbors(X, df['up_or_down'])
    # check_predictions(knn_predictions, knn_test, 'KNN')

    (regression_predictions, lm) = regression(X, y, X_train, X_test, y_train)
    check_predictions(regression_predictions, y_test, 'Linear Regresion')
    # print(_df.tail())
    _df.drop(target, axis=1, inplace=True)
    # print(_df.tail())
    print(_df[-6:])
    current_predictions = lm.predict(_df[-6:])
    print(current_predictions)
    # (k_means_cluster_pred, k_means_y_test) = run_k_means_cluster(
    #     X, df['up_or_down'])
    # check_predictions(k_means_cluster_pred, k_means_y_test, 'KNN')
    # Check the combined results?
    # check_combined(knn_predictions, regression_predictions, y_test)


def check_combined(knn_predictions, regression_predictions, y_test):
    for i, y in enumerate(y_test):
        print(knn_predictions[i], regression_predictions[i], y)


def k_nearest_neighbors(X, y):
    print(y)
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_features = scaler.transform(X)
    df_feat = pd.DataFrame(scaled_features, columns=X.columns)
    print(df_feat.head())
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y,
                                                        test_size=0.10)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(pred)
    print(y_test)
    return (pred, y_test)


def regression(X, y, X_train, X_test, y_train):
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_features = scaler.transform(X)
    df_feat = pd.DataFrame(scaled_features, columns=X.columns)
    print(df_feat.head())
    # X_train, X_test, y_train, y_test = train_test_split(scaled_features, y,
    #                                                     test_size=0.10)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    # print(lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)
    predictions = lm.predict(X_test)
    return (predictions, lm)


def check_predictions(predictions, y_test, name):
    print(f'Results for {name}')
    bothCorrect = 0
    accurate = 0
    for i, y in enumerate(y_test):
        p = predictions[i]
        # print(y, p)
        # print(y - p)
        # if (y <= 0 and p <= 0) or (y>= 0 and p >= 0):
        if (y == p) or (y < 0 and p < 0) or (y > 0 and p > 0):
            bothCorrect = bothCorrect+1
            if(abs(y - p) < 0.3):
                accurate = accurate + 1
    print(bothCorrect, accurate, len(predictions),
          bothCorrect/len(predictions)*100)


def run_k_means_cluster(X, y):
    # scaler = StandardScaler()
    # scaler.fit(X)
    # scaled_features = scaler.transform(X)
    # df_feat = pd.DataFrame(scaled_features, columns=X.columns)
    # print(df_feat.head())
    # X_train, X_test, y_train, y_test = train_test_split(scaled_features, y,
    #                                                     test_size=0.10, random_state=101)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.10, random_state=101)
    print(y_test)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_train)
    print(kmeans.cluster_centers_)
    pred = kmeans.predict(X_test)
    print(pred)
    # check results
    return (pred, y_test)
