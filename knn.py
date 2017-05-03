import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor

'''
Cleans the dataset:
    -Change format of close_date from string to datetime object
    -Imputes negative closing prices with the median close_price
    -Sorts the dataframe by increasing close_date so earlier close_dates always
     have a lower index than later close dates. This is to avoid the problem of
     temporal leakage (more details below)

INPUT: Dataframe
OUTPUT: Dataframe
'''
def clean_data(df):
    df['close_date'] = df['close_date'].apply(lambda x: datetime.datetime.\
                       strptime(x,'%M:%S.%f'))
    median_price = df['close_price'][df['close_price'] > 0].median()
    df['close_price'][df['close_price'] < 0] = median_price
    df.sort_values(by = 'close_date', inplace = True)
    df = df.reset_index(drop = True)
    return df

'''
Calculates the predicted closing price which is a function of:
    -A home's k-Nearest Neighbours as calculated by spatial distance
    -Weights of each of the nearest neighbours, with weights inv proportional
     to distance (closer points have more weight)
    -The training set always have close dates earlier than the testing set to
     prevent the issue of temporal leakage when making predictions

INPUT: Dataframe
OUTPUT: Dataframe
'''
def knn(df, num_splits, num_neighbors):
    tscv = TimeSeriesSplit(n_splits=num_splits)
    knn = KNeighborsRegressor(n_neighbors=num_neighbors, metric = 'euclidean', \
                              weights = 'distance')
    X = np.array(df[['latitude','longitude']])
    y = np.array(df['close_price'])
    Xtrain = []
    Xtest = []
    Ytrain = []
    Ytest = []
    medians = []
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Xtrain.append(X_train)
        Xtest.append(X_test)
        Ytrain.append(y_train)
        Ytest.append(y_test)
    for i in xrange(len(Xtrain)):
        knn.fit(Xtrain[i], Ytrain[i])
        pred = knn.predict(Xtest[i])
        med = np.median(abs(pred-Ytest[i])/Ytest[i])
        medians.append(med)
    return medians, np.mean(medians)

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df = clean_data(df)
    MRAE = knn(df,5,4)
    print MRAE
