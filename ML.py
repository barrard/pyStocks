import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def run_ML(df, target):
    print('Start of some ML?')
    df.dropna(inplace=True)
    # print(df)
    # print(df.describe())
    print(df.head())
    print(df.tail())

    y = df[target]
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    # (knn_predictions, knn_test) = k_nearest_neighbors(X, df['up_or_down'])
    # check_predictions(knn_predictions, knn_test, 'KNN')

    regression_predictions = regression(X, X_train, X_test, y_train)
    check_predictions(regression_predictions, y_test, 'Linear Regresion')

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
                                                        test_size=0.10, random_state=101)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(pred)
    print(y_test)
    return (pred, y_test)


def regression(X, X_train, X_test, y_train):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    # print(lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)
    predictions = lm.predict(X_test)
    return predictions


def check_predictions(predictions, y_test, name):
    print(f'Results for {name}')
    bothCorrect = 0
    accurate = 0
    for i, y in enumerate(y_test):
        p = predictions[i]
        print(y, p)
        print(y - p)
        if (y <= 0 and p <= 0) or (y>= 0 and p >= 0):
            bothCorrect = bothCorrect+1
            if(abs(y - p) < 0.3):
                accurate = accurate + 1
    print(bothCorrect, accurate, len(predictions),
          bothCorrect/len(predictions)*100)
