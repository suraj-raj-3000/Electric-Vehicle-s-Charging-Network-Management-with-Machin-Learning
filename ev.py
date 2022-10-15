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

from preprocess import *
from lib import *
from model import *

def json_to_DataFrame(file):
    with open(file) as data_file:
        data = json.load(data_file)
        df = pd.DataFrame(data["_items"])
        return df

caltech_df = json_to_DataFrame(file="./data/acndata_sessions_caltech.json")
print("\n\n\n ----- Caltech Data set is  -----\n\n",caltech_df.head())
jpl_df = json_to_DataFrame(file="./data/acndata_sessions_jpl.json")
print("\n\n\n ----- Jpl Data set is  -----\n\n",jpl_df.head())

# Checking if all the column names are same in both the DataFrames..
check = caltech_df.columns == jpl_df.columns

# Now we will compare the number of missing values and number of instances in each of the DataFrames..
df_info_dic = {"caltech_df":([caltech_df.shape[0]]+list(caltech_df.isnull().sum())),
"jpl_df":([jpl_df.shape[0]]+list(jpl_df.isnull().sum()))}
df_info = pd.DataFrame(df_info_dic, index=["total_instances"]+list(caltech_df.columns))

print("\n\nData Frame is : \n",df_info)


# Let us fill in all the missing values in userID column as unclaimed.
caltech_df["userID"] = caltech_df["userID"].fillna("unclaimed")
jpl_df["userID"] = jpl_df["userID"].fillna("unclaimed")

# Cross verify if all the missing values in userID columns are imputed or not..
caltech_df.userID.isnull().sum(), jpl_df.userID.isnull().sum()

# Let us replace all other instances in userID column as claimed..
caltech_df.loc[caltech_df["userID"]!="unclaimed", "userID"]="claimed"
jpl_df.loc[jpl_df["userID"]!="unclaimed", "userID"]="claimed"

# print(caltech_df.head())

# Number of unclaimed should be equal to number of missing values in the column.
# print(caltech_df.userID.value_counts()) 
# print(jpl_df.userID.value_counts())

df_info_dic = {"caltech_df":([caltech_df.shape[0]]+list(caltech_df.isnull().sum())),
"jpl_df":([jpl_df.shape[0]]+list(jpl_df.isnull().sum()))}
df_info = pd.DataFrame(df_info_dic, index=["total_instances"]+list(caltech_df.columns))

# print(df_info)

# print(caltech_df["clusterID"].value_counts())
# print(jpl_df["clusterID"].value_counts())

# Let us change the clusterID of each row for caltech_df to "0039"
caltech_df["clusterID"]="0039"
# print(caltech_df["clusterID"].value_counts())
# print(jpl_df["clusterID"].value_counts())

# print(caltech_df["siteID"].value_counts())
# print(jpl_df["siteID"].value_counts())

# Let us change the siteID of each row of caltech_df to "0002"
caltech_df["siteID"]="0002"
# print(caltech_df["siteID"].value_counts())

print("\n\nNumber of chargers in Caltech : ", len(caltech_df["spaceID"].value_counts()))
print("Number of chargers in JPL : ", len(jpl_df["spaceID"].value_counts()))

print("timezone caltech : ",caltech_df["timezone"].value_counts())
print("\ntimezone jpl : ",jpl_df["timezone"].value_counts())


# Now we will convert each instances of connectionTime, disconnectTime and doneChargingTime in datetime objects. Also
# we will convert the timezone from UTC to America/Los_Angeles Timezone.

print("\n\n\n------------------Done Charging Time -----------\n",caltech_df.iloc[4],caltech_df.iloc[:,3])
caltech_df = convert_datetime(caltech_df)
# print(caltech_df.head(5))

jpl_df = convert_datetime(jpl_df)

# print(caltech_df.head())
jpl_df_disconntime = jpl_df["disconnectTime"]


caltech_df["session_duration"] = (caltech_df["disconnectTime"] - caltech_df["connectionTime"])/timedelta(minutes=1)
jpl_df["session_duration"] = (jpl_df["disconnectTime"] - jpl_df["connectionTime"])/timedelta(minutes=1)
print("\n\n---- session_duration -----\n")
print("\n\n",caltech_df.head())

#Adding a Day column to both the DataFrames that signifies whether the EV was charged on a weekDay or a weekEnd
caltech_df["Day"] = caltech_df["connectionTime"].apply(lambda x : x.strftime("%a"))
caltech_df["Day"] = caltech_df["Day"].apply(lambda x : "weekEnd" if (x=="Sun" or x=="Sat") else "weekDay")

jpl_df["Day"] = jpl_df["connectionTime"].apply(lambda x : x.strftime("%a"))
jpl_df["Day"] = jpl_df["Day"].apply(lambda x : "weekEnd" if (x=="Sun" or x=="Sat") else "weekDay")

# Let us check the number of vehicles charged on weekDays compared to weekEnds..
print("caltech Day :\n",caltech_df["Day"].value_counts(normalize=True))

print(" \nJpl Day: \n",jpl_df["Day"].value_counts(normalize=True))

#TimeSeries analysis of each DataFrame
caltech_ts = caltech_df[["kWhDelivered"]]
caltech_ts.index = caltech_df["connectionTime"]
# print("\n ----- caltech_ts----- \n",caltech_ts.head())


# Now let us make a function to plot time series plots..
plt.figure(figsize=(10,7))
plot_time_series(caltech_ts[:50])


# Let us add a column as connectionDate in each dataframe..
caltech_df["connectionDate"] = caltech_df["connectionTime"].apply(lambda x : x.date)
jpl_df["connectionDate"] = jpl_df["connectionTime"].apply(lambda x : x.date)

# Now let us make a function that calculates total energy consumed and total sessions served on a single day.
caltech_ts = make_correct_time_series(caltech_df)
jpl_ts = make_correct_time_series(jpl_df)

# print("----------Jpl------\n",jpl_ts.head(5))




#-------Understanding User Behaviour--------
user_behaviour(caltech_ts)
jpl_energy_demand(jpl_ts)
caltech_evse(caltech_ts)
jpl_evse(jpl_ts)







#----------------------------

caltech_new = caltech_df.copy()
jpl_new = jpl_df.copy()
caltech_new["connectionTime"] = caltech_new["connectionTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))
caltech_new["disconnectTime"] = caltech_new["disconnectTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))
# caltech_new["doneChargingTime"] = caltech_new["doneChargingTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))

jpl_new["connectionTime"] = jpl_new["connectionTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))
jpl_new["disconnectTime"] = jpl_new["disconnectTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))
# jpl_new["doneChargingTime"] = jpl_new["doneChargingTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))

caltech_new["connectionMonth"] = caltech_new["connectionDate"].apply(lambda x : x.strftime("%b"))
print("\n\n",caltech_new.head())

# Plotting arrival time for Free and Paid users on weekDays and weekEnds
a1 = caltech_new.loc[(caltech_new["userID"]=="unclaimed")&(caltech_new["Day"]=="weekDay")]["connectionTime"]
a2 = caltech_new.loc[(caltech_new["userID"]=="claimed")&(caltech_new["Day"]=="weekDay")]["connectionTime"]
a3 = caltech_new.loc[(caltech_new["userID"]=="unclaimed")&(caltech_new["Day"]=="weekEnd")]["connectionTime"]
a4 = caltech_new.loc[(caltech_new["userID"]=="claimed")&(caltech_new["Day"]=="weekEnd")]["connectionTime"]

fre_paid_charging(a1,a2,a3,a4)


# # ========== Model =================

# Let us add a column as connectionDate in each dataframe..
df = pd.concat([caltech_df, jpl_df], axis=0)
print("\n",df.head())

simple_df = df.loc[:, ["connectionTime", "disconnectTime", "kWhDelivered"]]

# ---------calling linear regration -----
linear_regration(simple_df)

# Let us plot a scatter plot to furthur understand the dataset..
kWhDelivered_plot(simple_df)


    # ----------- Analysis of the scatter plot -------------
session_length = list(simple_df["session_length"])
session_length[:10]
session_len_copied = session_length.copy()

    # Let us sort the list in ascending order
session_len_copied.sort()

    # Analysing to 10 smallest session lengths in the dataframe.

print('''\n\n\nHere EV at index number 246 has been charged for around 1 minute but has consumed 0.586 kWh of energy. It seems there is some problem here.
The Ev was connected at 11:45 AM and disconnected at 4:22 PM but its battery became fully charged at 11:46 AM''')


print("\n\nsession_len_copied \n",session_len_copied[:50],end="\n")

print("\nsession_length : ",session_length.index(6.966666666666667))
d1 = simple_df.copy()
print("\n",d1.iloc[3648])

print("\n",caltech_df.iloc[3648])




plt.figure(figsize=(10,10))
simple_df[["session_length"]].boxplot()
plt.show()

for x in ['session_length']:
    q75,q25 = np.percentile(simple_df.loc[:,x],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    simple_df.loc[simple_df[x] < min,x] = np.nan
    simple_df.loc[simple_df[x] > max,x] = np.nan

print(simple_df["session_length"].isnull().sum())

print("\n\nHence there are 1812 outliers in the session length column of the dataframe. We have to remove these rows")

simple_df = simple_df.dropna()
simple_df["session_length"].isnull().sum()

print("\n",simple_df.shape)
# Now let us find the correlation again.
correlation = simple_df.corr()
print("\n\n\n",correlation)

print('''\n\n\n The correlation between kWhDelivered and session_length columns was around 48% before the removal of outliers has been improved to 60% after
the removal of the outliers. This increment is significant ''')

scatter_plot_energy(simple_df)




# ----------- Splitting the dataset into a train and test set------------
print("\n\nSplitting the dataset into a train and test set")
print(simple_df.shape)

# Shuffle the dataframe
simple_df = simple_df.sample(frac=1, random_state=42)

X = simple_df[["session_length"]]
y = simple_df[["kWhDelivered"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(simple_df.head())

# ------------ Model 1 : Linear Regression -------------------
print("\n\nModel 1 : Linear Regression\n\n")
model_1_lr = LinearRegression()
# Fit the training data into the model..
model_1_lr.fit(X_train, y_train)
# Our model has been trained. Let us predict on test datasets.
y_pred = model_1_lr.predict(X_test)
model_1_performance = calculate_performance(y_test, y_pred)
print(model_1_performance)


# ----------- Model 2 : Random Forest Regresson ----------------
print("\n\nModel 2 : Random Forest Regresson\n")
model_2_rf = RandomForestRegressor()
model_2_rf.fit(X_train, y_train)
y_pred_rf = model_2_rf.predict(X_test)


model_2_performance = calculate_performance(y_test, y_pred_rf)
print(model_2_performance)

print(y_test[:10], y_pred_rf[:10])
# Using Cross validation to train the Random Forest Model
print("\n\nUsing Cross validation to train the Random Forest Model\n")
scores = cross_val_score(model_2_rf, X, y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
rmse = np.mean(rmse_scores)
print("\n\n",rmse)

# ------------- Model 3 : Support Vector Machine ---------------------
print("\n\nModel 3 : Support Vector Machine \n")
model_3_svr = SVR()
model_3_svr.fit(X_train, y_train)
y_pred_svr = model_3_svr.predict(X_test)

model_3_performance = calculate_performance(y_test, y_pred_svr)
print(model_3_performance)

# --------------------Model 4 : XGBoost -----------------------
from xgboost import XGBRegressor
print("\n\nModel 4 : XGBoost\n")
model_4_xgb = XGBRegressor()
model_4_xgb.fit(X_train, y_train)
y_pred_xgb = model_4_xgb.predict(X_test)
model_4_performance = calculate_performance(y_test, y_pred_xgb)
print(model_4_performance)

# -------------------Comparing the results of all 4 models--------------
print("\n\n\n\n-------Comparing the results of all 4 models--------")
results_dic = {"Linear Regression" : model_1_performance,"Random Forest" : model_2_performance,"Support Vector Machines" : model_3_performance,"XGBoost" : model_4_performance}
results_df = pd.DataFrame(results_dic)
print(results_df)
print("\n\n\n")










