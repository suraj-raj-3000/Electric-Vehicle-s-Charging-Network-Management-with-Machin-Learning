import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import tz
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

def linear_regration(simple_df):
    d1 = simple_df.copy()
    print(simple_df.head())

    # Now add a column, "session_length" in the dataframe.
    simple_df["session_length"] = (simple_df["disconnectTime"] - simple_df["connectionTime"])/timedelta(minutes=1)
    print(simple_df)
    # drop "connectionTime" and "doneChargingTime" columns..
    simple_df = simple_df.drop(columns=["connectionTime", "disconnectTime"])
    print(simple_df.head())
    # Check the correlation..
    correlation = simple_df.corr()
    print(correlation)


# Build a function to calculate the model performance

def calculate_performance(y_test, y_pred):
    MAE = np.round(mean_absolute_error(y_test, y_pred),2)
    RMSE = np.round(np.sqrt(mean_squared_error(y_test, y_pred)),2)
    r_sq = np.round(r2_score(y_test, y_pred),2)
    performance_dic = {"MAE": MAE, "RMSE" : RMSE, "r2_score" : r_sq}
    return performance_dic

