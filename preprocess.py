import json
import pandas as pd
import numpy as np
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

# def gethering_data(df):
#     # df = df.dropna()
#     df_train=pd.DataFrame(df,columns= ['selftext','ANNOTATIONS'])
#     df_train = df_train.rename(columns={'selftext': 'text','ANNOTATIONS':'label'})
#     df_train= df_train.convert_dtypes()
#     print(df_train)

def convert_datetime(df):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Los_Angeles')
    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x : datetime.strptime(x, "%a, %d %b %Y %H:%M:%S %Z"))
    df.iloc[:, 3] = df.iloc[:, 3].apply(lambda x : datetime.strptime(x, "%a, %d %b %Y %H:%M:%S %Z"))

    df.iloc[:,2] = df.iloc[:, 2].apply(lambda x : x.replace(tzinfo=from_zone))
    df.iloc[:,3] = df.iloc[:, 3].apply(lambda x : x.replace(tzinfo=from_zone))
    # df.iloc[:,4] = df.iloc[:, 4].apply(lambda x : x.replace(tzinfo=from_zone))

    df.iloc[:,2] = df.iloc[:, 2].apply(lambda x : x.astimezone(to_zone))
    df.iloc[:,3] = df.iloc[:, 3].apply(lambda x : x.astimezone(to_zone))
    # df.iloc[:,4] = df.iloc[:, 4].apply(lambda x : x.astimezone(to_zone))

    return df

def make_correct_time_series(df):
    list1 = list(df["connectionDate"])
    list2 = list(set(list1))
    list2.sort()
    indices_list = []
    for i in list2:
        indices_list.append(list1.index(i))
    sessions_served_each_day = []
    for i in list2:
        sessions_served_each_day.append(list1.count(i))
    # Calculate energy demand on a specific day..
    energy_demand_per_day = []
    for i, j in zip(indices_list, sessions_served_each_day):
        energy_demand = []
        for x in df.iloc[i:(i+j), 5]:
            energy_demand.append(x)
        energy_demand_per_day.append(np.round(np.sum(np.array(energy_demand)),2))
    # Now make a dict of connectionDate, energyDemand and sessions.
    df_dic = {"connectionDate":list2,"energyDemand" : energy_demand_per_day,"sessions" : sessions_served_each_day}
    # Now convert this dictionary into a DataFrame.
    ts = pd.DataFrame(df_dic)
    # make connectionDate as the index of the DataFrame.
    ts = ts.set_index(["connectionDate"])
    return ts


def mtn(x):
    months = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
         'may':5,
         'jun':6,
         'jul':7,
         'aug':8,
         'sep':9,
         'oct':10,
         'nov':11,
         'dec':12
        }
    c=months[x]
    return c
    
s="wed, 25 apr 2018 13:21:10 GMT"
a=s[5:]
b=a[:-13]

for i in range(len(b)):
    if b[i].isalpha():
        ch=b[i:i+3]
        mn=mtn(str(ch))
        mnn=str(mn)
        b=b[:i]+mnn+b[i+3:]
        break
print(b)