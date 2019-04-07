import os
os.chdir(r"C:\Users\yliu10\Desktop\Delaware")
import random
random.seed (2)

import pandas as pd
import numpy as np

#EPEX
df = pd.read_csv("EPEX_combined.csv")
del df['Unnamed: 0']
df=df.rename(columns = {'d_DateTime':'date_time'})

#Weather
bordeaux = pd.read_csv("Bordeaux.csv")
dijon = pd.read_csv("Dijon.csv")
lille = pd.read_csv("lille.csv")
lyon = pd.read_csv("lyon.csv")
marseille = pd.read_csv("MARSEILLE.csv")
montpellier = pd.read_csv("MONTPELLIER.csv")
poitiers = pd.read_csv("POITIERS.csv")
remis = pd.read_csv("reim.csv")

del [bordeaux['Unnamed: 0'], dijon['Unnamed: 0'], lille['Unnamed: 0'], lyon['Unnamed: 0'],
     marseille['Unnamed: 0'], montpellier['Unnamed: 0'], poitiers['Unnamed: 0'], remis['Unnamed: 0']]

df_1 = pd.merge(df,bordeaux,on='date_time', how='left')
df_2 = pd.merge(df_1,dijon,on='date_time', how='left')
df_3 = pd.merge(df_2,lille,on='date_time', how='left')
df_4 = pd.merge(df_3,lyon,on='date_time', how='left')
df_5 = pd.merge(df_4,marseille,on='date_time', how='left')
df_6 = pd.merge(df_5,montpellier,on='date_time', how='left')
df_7 = pd.merge(df_6,poitiers,on='date_time', how='left')
epex_weather = pd.merge(df_7,remis,on='date_time', how='left')

del [df, bordeaux, dijon, lille, lyon, marseille, montpellier, poitiers, remis,
     df_1, df_2, df_3, df_4, df_5, df_6, df_7]

#Reanme
epex_weather.columns = ['date_time',
 'd_Prices',
 'd_Volume',
 'd_MiddleNight',
 'd_EarlyMorning',
 'd_LateMorning',
 'd_EarlyAfternoon',
 'd_RushHour',
 'd_OffPeak2',
 'd_Night',
 'd_OffPeak1',
 'd_Business',
 'd_OffPeak',
 'd_Morning',
 'd_HighNoon',
 'd_Afternoon',
 'd_Evening',
 'd_SunPeak',
 'd_BasePrice',
 'd_BaseVolume',
 'd_PeakPrice',
 'd_PeakVolume',
 'i_DateTime',
 'i_Low',
 'i_High',
 'i_Last',
 'i_Weighted_Avg',
 'i_Idx',
 'i_ID3',
 'i_Buy_Vol',
 'i_Sell_Vol',
 'i_Index_Base',
 'i_Index_Peak',
 'bordeaux_maxtempC',
 'bordeaux_mintempC',
 'bordeaux_totalSnow_cm',
 'bordeaux_sunHour',
 'bordeaux_uvIndex',
 'bordeaux_moon_illumination',
 'bordeaux_moonrise',
 'bordeaux_moonset',
 'bordeaux_sunrise',
 'bordeaux_sunset',
 'bordeaux_DewPointC',
 'bordeaux_FeelsLikeC',
 'bordeaux_HeatIndexC',
 'bordeaux_WindChillC',
 'bordeaux_WindGustKmph',
 'bordeaux_cloudcover',
 'bordeaux_humidity',
 'bordeaux_precipMM',
 'bordeaux_pressure',
 'bordeaux_tempC',
 'bordeaux_visibility',
 'bordeaux_winddirDegree',
 'bordeaux_windspeedKmph',
 'dijon_maxtempC',
 'dijon_mintempC',
 'dijon_totalSnow_cm',
 'dijon_sunHour',
 'dijon_uvIndex',
 'dijon_moon_illumination',
 'dijon_moonrise',
 'dijon_moonset',
 'dijon_sunrise',
 'dijon_sunset',
 'dijon_DewPointC',
 'dijon_FeelsLikeC',
 'dijon_HeatIndexC',
 'dijon_WindChillC',
 'dijon_WindGustKmph',
 'dijon_cloudcover',
 'dijon_humidity',
 'dijon_precipMM',
 'dijon_pressure',
 'dijon_tempC',
 'dijon_visibility',
 'dijon_winddirDegree',
 'dijon_windspeedKmph',
 'lille_maxtempC',
 'lille_mintempC',
 'lille_totalSnow_cm',
 'lille_sunHour',
 'lille_uvIndex',
 'lille_moon_illumination',
 'lille_moonrise',
 'lille_moonset',
 'lille_sunrise',
 'lille_sunset',
 'lille_DewPointC',
 'lille_FeelsLikeC',
 'lille_HeatIndexC',
 'lille_WindChillC',
 'lille_WindGustKmph',
 'lille_cloudcover',
 'lille_humidity',
 'lille_precipMM',
 'lille_pressure',
 'lille_tempC',
 'lille_visibility',
 'lille_winddirDegree',
 'lille_windspeedKmph',
 'lyon_maxtempC',
 'lyon_mintempC',
 'lyon_totalSnow_cm',
 'lyon_sunHour',
 'lyon_uvIndex',
 'lyon_moon_illumination',
 'lyon_moonrise',
 'lyon_moonset',
 'lyon_sunrise',
 'lyon_sunset',
 'lyon_DewPointC',
 'lyon_FeelsLikeC',
 'lyon_HeatIndexC',
 'lyon_WindChillC',
 'lyon_WindGustKmph',
 'lyon_cloudcover',
 'lyon_humidity',
 'lyon_precipMM',
 'lyon_pressure',
 'lyon_tempC',
 'lyon_visibility',
 'lyon_winddirDegree',
 'lyon_windspeedKmph',
 'marseille_maxtempC',
 'marseille_mintempC',
 'marseille_totalSnow_cm',
 'marseille_sunHour',
 'marseille_uvIndex',
 'marseille_moon_illumination',
 'marseille_moonrise',
 'marseille_moonset',
 'marseille_sunrise',
 'marseille_sunset',
 'marseille_DewPointC',
 'marseille_FeelsLikeC',
 'marseille_HeatIndexC',
 'marseille_WindChillC',
 'marseille_WindGustKmph',
 'marseille_cloudcover',
 'marseille_humidity',
 'marseille_precipMM',
 'marseille_pressure',
 'marseille_tempC',
 'marseille_visibility',
 'marseille_winddirDegree',
 'marseille_windspeedKmph',
 'montpellier_maxtempC',
 'montpellier_mintempC',
 'montpellier_totalSnow_cm',
 'montpellier_sunHour',
 'montpellier_uvIndex',
 'montpellier_moon_illumination',
 'montpellier_moonrise',
 'montpellier_moonset',
 'montpellier_sunrise',
 'montpellier_sunset',
 'montpellier_DewPointC',
 'montpellier_FeelsLikeC',
 'montpellier_HeatIndexC',
 'montpellier_WindChillC',
 'montpellier_WindGustKmph',
 'montpellier_cloudcover',
 'montpellier_humidity',
 'montpellier_precipMM',
 'montpellier_pressure',
 'montpellier_tempC',
 'montpellier_visibility',
 'montpellier_winddirDegree',
 'montpellier_windspeedKmph',
 'poitiers_maxtempC',
 'poitiers_mintempC',
 'poitiers_totalSnow_cm',
 'poitiers_sunHour',
 'poitiers_uvIndex',
 'poitiers_moon_illumination',
 'poitiers_moonrise',
 'poitiers_moonset',
 'poitiers_sunrise',
 'poitiers_sunset',
 'poitiers_DewPointC',
 'poitiers_FeelsLikeC',
 'poitiers_HeatIndexC',
 'poitiers_WindChillC',
 'poitiers_WindGustKmph',
 'poitiers_cloudcover',
 'poitiers_humidity',
 'poitiers_precipMM',
 'poitiers_pressure',
 'poitiers_tempC',
 'poitiers_visibility',
 'poitiers_winddirDegree',
 'poitiers_windspeedKmph',
 'reim_maxtempC',
 'reim_mintempC',
 'reim_totalSnow_cm',
 'reim_sunHour',
 'reim_uvIndex',
 'reim_moon_illumination',
 'reim_moonrise',
 'reim_moonset',
 'reim_sunrise',
 'reim_sunset',
 'reim_DewPointC',
 'reim_FeelsLikeC',
 'reim_HeatIndexC',
 'reim_WindChillC',
 'reim_WindGustKmph',
 'reim_cloudcover',
 'reim_humidity',
 'reim_precipMM',
 'reim_pressure',
 'reim_tempC',
 'reim_visibility',
 'reim_winddirDegree',
 'reim_windspeedKmph']

epex_weather['date_time'] = pd.to_datetime(epex_weather['date_time'])

#Entose
forecast_transfer_capacities = pd.read_csv("Final_Data_Forecasted_Transfer_Capacities_day_ahead.csv")
forecast_transfer_capacities.drop(forecast_transfer_capacities.index[43876], inplace=True)
forecast_transfer_capacities['fromtime'] = (forecast_transfer_capacities['fromtime']*60).astype(int)
forecast_transfer_capacities['fromtime'] = pd.to_datetime(forecast_transfer_capacities.fromtime, unit='m').dt.strftime('%H:%M')
forecast_transfer_capacities['date_time'] = forecast_transfer_capacities['fromdate'] + " " + forecast_transfer_capacities['fromtime']
forecast_transfer_capacities.drop(['fromdate'], axis=1, inplace=True)
forecast_transfer_capacities.drop(['fromtime'], axis=1, inplace=True)
forecast_transfer_capacities['date_time'] = pd.to_datetime(forecast_transfer_capacities['date_time'])
cols = forecast_transfer_capacities.columns.tolist()
cols = cols[-1:] + cols[:-1]
forecast_transfer_capacities = forecast_transfer_capacities[cols]
columntype = forecast_transfer_capacities.dtypes
forecast_transfer_capacities = forecast_transfer_capacities.sort_values(['date_time'])
epex_weather_entose = pd.merge(epex_weather, forecast_transfer_capacities ,on='date_time', how='left')
del [forecast_transfer_capacities, columntype, cols, epex_weather]


actual = pd.read_csv("Final_Day_ahead_Actual.csv")
actual['fromtime'] = (actual['fromtime']*60).astype(int)
actual['fromtime'] = pd.to_datetime(actual.fromtime, unit='m').dt.strftime('%H:%M')
actual['date_time'] = actual['fromdate'] + " " + actual['fromtime']
actual.drop(['fromdate'], axis=1, inplace=True)
actual.drop(['fromtime'], axis=1, inplace=True)
actual['date_time'] = pd.to_datetime(actual['date_time'])
cols = actual.columns.tolist()
cols = cols[-1:] + cols[:-1]
actual = actual[cols]
epex_weather_entose = pd.merge(epex_weather_entose, actual ,on='date_time', how='left')
del [actual, cols]


price_transmission = pd.read_csv("Final_Day_Ahead_Prices_Transmission.csv")
price_transmission['fromtime'] = (price_transmission['fromtime']*60).astype(int)
price_transmission['fromtime'] = pd.to_datetime(price_transmission.fromtime, unit='m').dt.strftime('%H:%M')
price_transmission['date_time'] = price_transmission['fromdate'] + " " + price_transmission['fromtime']
price_transmission.drop(['fromdate'], axis=1, inplace=True)
price_transmission.drop(['fromtime'], axis=1, inplace=True)
price_transmission['date_time'] = pd.to_datetime(price_transmission['date_time'])
epex_weather_entose = pd.merge(epex_weather_entose, price_transmission ,on='date_time', how='left')
del [price_transmission]


holiday = pd.read_csv("France_Holidays_combined.csv")
del(holiday['Unnamed: 0'])
holiday['date_time']= pd.to_datetime(holiday['date_time'])
df = pd.merge(epex_weather_entose,holiday,on='date_time', how='left')

del [epex_weather_entose]
#df.to_csv("df_merged.csv")
#df = df[df.date_time.between('2015-01-01','2018-12-31')]
#columntype = df.dtypes
#describe = np.round(df.describe(), 2).T
#describe.to_csv("describe.csv")
#corr = df.corr()
#corr.to_csv("corr.csv")
#sns.boxplot(x=df["Prices"],data=df)
#list(df.columns.values)


""" Drop variables """

drop = ["cloudcover", "uvIndex", "moonrise","moonset",
        "sunrise", "sunset", "sunHour"]
locate = ['bordeaux_', 'dijon_', 'lille_', 'lyon_','marseille_', 'montpellier_',
          'poitiers_', 'reim_']
for variable in drop:
    for area in locate:
        df.drop([area + variable], axis=1, inplace=True)
del [drop, locate, area, variable]

df.drop(['i_DateTime',
         'i_Low',
         'i_High',
         'i_Last',
         'i_Weighted_Avg',
         'i_ID3',
         'i_Buy_Vol',
         'i_Sell_Vol',
         'i_Index_Base',
         'i_Index_Peak',
         'AL_MW',
         'GR_MW',
         'Dayahead_Load Forecast',
         'ActualTotalLoad',
         'DayaheadPrice _EUR_MWh',
         'd_MiddleNight',
         'd_EarlyMorning',
         'd_LateMorning',
         'd_EarlyAfternoon',
         'd_RushHour',
         'd_OffPeak2',
         'd_Night',
         'd_OffPeak1',
         'd_Business',
         'd_OffPeak',
         'd_Morning',
         'd_HighNoon',
         'd_Afternoon',
         'd_Evening',
         'd_SunPeak',
         'd_BasePrice',
         'd_BaseVolume',
         'd_PeakPrice',
         'd_PeakVolume',
         'd_Volume'], axis=1,inplace=True)


""" Create variables """
df[['date_time','d_Prices']][df['d_Prices'].isnull()]

df['price_diff'] = df['d_Prices']-df['i_Idx']

#Target to 1 or 0
df['d_higher'] = np.where(df['price_diff']>0, 1,0)

#variable = "FeelsLikeC"
#dates = pd.to_datetime(epex_weather['date_time'])
#plt.figure(figsize=(15,10))
#plt.title(variable)
#plt.plot_date(dates, epex_weather['bordeaux_'+ variable])
#plt.plot_date(dates, epex_weather['dijon_'+ variable])
#plt.plot_date(dates, epex_weather['lille_'+ variable])
#plt.plot_date(dates, epex_weather['lyon_'+ variable])
#plt.plot_date(dates, epex_weather['marseille_'+ variable])
#plt.plot_date(dates, epex_weather['montpellier_'+ variable])
#plt.plot_date(dates, epex_weather['poitiers_'+ variable])
#plt.plot_date(dates, epex_weather['reim_'+ variable])
#plt.show
#epex_weather['average_'+ variable] = epex_weather[['bordeaux_'+ variable, 'dijon_'+ variable
#            , 'lille_'+ variable, 'lyon_'+ variable, 'marseille_'+ variable
#            , 'montpellier_'+ variable, 'poitiers_'+ variable, 'reim_'+ variable]].mean(axis=1)
#epex_weather.drop(['bordeaux_'+ variable, 'dijon_'+ variable
#            , 'lille_'+ variable, 'lyon_'+ variable, 'marseille_'+ variable
#            , 'montpellier_'+ variable, 'poitiers_'+ variable, 'reim_'+ variable], axis=1, inplace=True)
#list(epex_weather)

#Combine and drop weather 
average = ["DewPointC", "FeelsLikeC", "HeatIndexC", "humidity", "maxtempC", "mintempC",
           "moon_illumination", "totalSnow_cm", "WindChillC", "tempC", "WindGustKmph",
           "windspeedKmph", "pressure", "visibility", "precipMM", "winddirDegree"]

for variable in average:
    df['average_'+ variable] = df[['bordeaux_'+ variable,
                       'dijon_'+ variable, 'lille_'+ variable
                       , 'lyon_'+ variable, 'marseille_'+ variable
                       , 'montpellier_'+ variable, 'poitiers_'+ variable
                       , 'reim_'+ variable]].mean(axis=1)
    df.drop(['bordeaux_'+ variable,
                       'dijon_'+ variable, 'lille_'+ variable
                       , 'lyon_'+ variable, 'marseille_'+ variable
                       , 'montpellier_'+ variable, 'poitiers_'+ variable
                       , 'reim_'+ variable], axis=1, inplace=True)
del [average, variable]


import seaborn as sns
import matplotlib.pyplot as plt 
import os

df['date_time']=pd.to_datetime(df['date_time'])
df.index = df['date_time']
df.shape
df.info()
df.isnull().sum()
list(df)

from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
rcParams['figure.figsize'] = 11, 6


### drop missing price diff rows for now
result = seasonal_decompose(df['price_diff'].dropna(), model='addtive', freq=365*24)
fig = result.plot()
plt.show()

# Date features
df['Hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek 
df['dayofyear'] = df.index.dayofyear
df['weekofyear'] = df.index.weekofyear
df['Month'] = df.index.month
df['quarter'] = df.index.quarter

#df['Last24hrs_mean'] = df['price_diif'].rolling(24).mean().shift(24)

#top100_diff = df.nlargest(100, 'price_diff') # top 5 highest
#bottom100_diff = df.nsmallest(100, 'price_diff')

## Lag data (previous hours)
def add_Lag(df, lags):
    for i in lags:
        df[f'lag_{i}'] = df['price_diff'].shift(i)
    return df

df = add_Lag(df,[24,48,72,96,120,144,168,744,2196,8760]) # [24,48,72,96,120,144,168] , range(1,25)

pv = pd.pivot_table(df, index=df.index.dayofyear, columns=df.index.year,
                    values='price_diff', aggfunc='mean')
pv.plot(title="price_diff Daily Average Yearly comparison ",figsize=(14,6), grid=True)
# month 1-3, 11-12 has many peaks > 10 EURs

pv2 = pd.pivot_table(df, index=df.index.dayofyear, columns=df.index.year,
                    values='price_diff', aggfunc='max')
pv2.plot(title="price_diff Daily Maximum Yearly comparison ",figsize=(14,6), grid=True)

peak_months = df.loc[(df['Month'] >= 11) | (df['Month'] <= 3)]
fig, ax = plt.subplots(figsize=(6,4)) 
hrvsday = peak_months.pivot_table(values='price_diff',index='Hour',columns='dayofweek',aggfunc='mean')
ax.set_title('price_diff "Day of week vs. Hour" in average')
sns.heatmap(hrvsday,cmap='magma_r', ax=ax) #  Monday=0, Sunday=6
ax.set_xlabel('dayofweek (Mon=0, Sun=6)')
plt.show()



""" basetable plots """
#Price fluctuation
years = [2014,2015,2016,2017,2018]
for i in years:
    x = df[df['date_time'].dt.year == i]
    plt.figure(figsize=(6,4))
    plt.plot(x['date_time'], x['d_Prices'])
    plt.plot(x['i_Idx'], 'r', alpha=0.3)
    plt.title(i)
    plt.legend(loc="upper right", shadow=True, fancybox =True) 
    plt.grid(True)
    plt.show()


# Target around 50% each year
years = [2014,2015,2016,2017,2018]
for i in years:
    plt.figure(figsize=(6,4))
    x = df[df['date_time'].dt.year == i]
    sns.countplot(x['d_higher'])
    plt.title(i)
    plt.show()



features = ["dayofweek","Month","quarter","dayofyear"]
for i in features:
    fig, ax = plt.subplots(figsize=(6,4))
    hrvsday = df.pivot_table(values='d_higher',index='Hour',columns=i,aggfunc='mean')
    ax.set_title("d_higher " + i + " vs. Hour in average")
    sns.heatmap(hrvsday,cmap='magma_r', ax=ax) #  Monday=0, Sunday=6
    plt.show()


features = ['average_DewPointC',
             'average_FeelsLikeC',
             'average_HeatIndexC',
             'average_humidity',
             'average_maxtempC',
             'average_mintempC',
             'average_moon_illumination',
             'average_totalSnow_cm',
             'average_WindChillC',
             'average_tempC',
             'average_WindGustKmph',
             'average_windspeedKmph',
             'average_pressure',
             'average_visibility',
             'average_precipMM', 
             "average_winddirDegree"]
for i in features:
    x = df.groupby([i])[['d_higher']].mean()
    plt.figure(figsize=(6,4))
    plt.title(i)
    plt.plot(x.index, x['d_higher'])
    #plt.axis([0, 6, 0, 20])
    plt.legend(loc="lower right", shadow=True, fancybox =True) 
    plt.grid(True)
    plt.show()
    del [x]



""" Finalize basetable for use"""
# change dayofweek to dummies 
df.dayofweek = df.dayofweek.apply(str)
df = pd.get_dummies(df, prefix=['dayofweek'])

df = df.dropna()
df.info()
# drop date_time column and prices that we do not know in advance
#df.drop(['date_time','d_Prices','i_Idx','price_diff'], axis=1,inplace=True)


"""Adjust feature"""
df.drop([
 'isHoliday'
 ], axis=1,inplace=True)


"""## Split train/test data"""
train_percent = 0.7
train = df.iloc[:int(len(df)*train_percent)]
test = df.iloc[int(len(df)*train_percent):]
X_train = train.drop(columns=['d_higher'])
X_test = test.drop(columns=['d_higher'])
y_train = train['d_higher']
y_test = test['d_higher']
y_test[y_test.isnull()]
X_train.drop(['date_time','d_Prices','i_Idx','price_diff'], axis=1,inplace=True)
X_test.drop(['date_time','d_Prices','i_Idx','price_diff'], axis=1,inplace=True)
sum(test['price_diff'])

# over sampling train data
#from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(random_state=0)
#X_train, y_train = ros.fit_resample(X_train, y_train)
#X_train = pd.DataFrame(data=X_train)
#X_train.columns = list(X_test)


""" DecisionTree Classifier """
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', 
                                  max_depth=4, min_samples_split=2, 
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                  max_features=None, random_state=None, 
                                  max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, class_weight=None, 
                                  presort=False)
clf.fit(X_train, y_train)
clf_train_pred = clf.predict(X_train)
clf_test_pred = clf.predict(X_test)

predictors=list(X_train)
feat_imp = pd.Series(clf.feature_importances_, predictors).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Importance of Features')


""" Evaluation """
from sklearn.metrics import roc_auc_score, roc_curve, auc,mean_squared_error\
                            ,mean_absolute_error, confusion_matrix\
                            ,classification_report, accuracy_score

cm = confusion_matrix(y_test,clf_test_pred)
print("confusion_matrix: \n",cm)
total=sum(sum(cm))
sensitivity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Specificity : ', specificity1)
accuracy=(cm[0,0]+cm[1,1])/total
print ('Accuracy : ', accuracy)

test['predict'] = clf_test_pred
print(test.groupby(['d_higher'])[["price_diff"]].sum())
print(test.groupby(['d_higher','predict'])[["price_diff"]].sum())


describe = np.round(test.describe(), 2).T

#print("classification_report: \n",classification_report(y_test,clf_test_pred))
#print("train_roc_auc_score:",roc_auc_score(y_train, clf_train_pred))
#print("test_roc_auc_score:",roc_auc_score(y_test, clf_test_pred))
#print("train_accuracy_score:",accuracy_score(y_train, clf_train_pred))
#print("test_accuracy_score:",accuracy_score(y_test, clf_test_pred))
#print("train_mean_squared_error:",mean_squared_error(y_train, clf_train_pred))
#print("test_mean_squared_error:",mean_squared_error(y_test, clf_test_pred))
#print("train_mean_absolute_error:",mean_absolute_error(y_train, clf_train_pred))
#print("test_mean_absolute_error:",mean_absolute_error(y_test, clf_test_pred))

#y_score = clf.predict_proba(X_train)[:,1]
#train_fpr = dict()
#train_tpr = dict()
#train_fpr, train_tpr, _ = roc_curve(y_train, y_score)
#train_roc_auc = dict()
#train_roc_auc = auc(train_fpr, train_tpr)
#print("train_roc_auc:",train_roc_auc)

#y_score = clf.predict_proba(X_test)[:,1]
#fpr = dict()
#tpr = dict()
#fpr, tpr, _ = roc_curve(y_test, y_score)
#roc_auc = dict()
#roc_auc = auc(fpr, tpr)
#print("test_roc_auc:",roc_auc)

# make the plot
#plt.figure(figsize=(5,5))
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.grid(True)
#plt.plot(train_fpr, train_tpr, label='Train_AUC = {0}'.format(train_roc_auc)) 
#plt.plot(fpr, tpr, label='Test_AUC = {0}'.format(roc_auc))         
#plt.legend(loc="lower right", shadow=True, fancybox =True) 
#plt.show()
#del [fpr, tpr, roc_auc, y_score]