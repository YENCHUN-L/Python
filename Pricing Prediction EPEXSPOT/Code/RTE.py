import os
import pandas as pd
import numpy as np
os.chdir(r"C:\Users\yliu10\Desktop\Delaware\RTE\Generation_Forecast\Solar")
PrevisionSolaire_2014 = pd.read_csv("PrevisionSolaire_2014.csv")
PrevisionSolaire_2015 = pd.read_csv("PrevisionSolaire_2015.csv")
PrevisionSolaire_2016 = pd.read_csv("PrevisionSolaire_2016.csv")
PrevisionSolaire_2017 = pd.read_csv("PrevisionSolaire_2017.csv")
PrevisionSolaire_2018 = pd.read_csv("PrevisionSolaire_2018.csv")
solar = pd.concat([PrevisionSolaire_2014, PrevisionSolaire_2015,
                PrevisionSolaire_2016, PrevisionSolaire_2017,
                PrevisionSolaire_2018])
del [PrevisionSolaire_2014, PrevisionSolaire_2015,
                PrevisionSolaire_2016, PrevisionSolaire_2017,
                PrevisionSolaire_2018]
solar['date_time'] = solar['Date'] + " " + solar['Heure']
solar.drop(['Date', 'Heure'], axis=1, inplace=True)
solar['date_time'] = pd.to_datetime(solar['date_time'])
solar.dtypes
solar['Prévision J'] = np.where(solar['Prévision J']=="-",0,solar['Prévision J'])
solar['Prévision J-1'] = np.where(solar['Prévision J-1']=="-",0,solar['Prévision J-1'])
solar['Prévision J'] = solar['Prévision J'].astype(str).astype(float)
solar['Prévision J-1'] = solar['Prévision J-1'].astype(str).astype(float)
#df = pd.merge(df,solar,on='date_time', how='left')



os.chdir(r"C:\Users\yliu10\Desktop\Delaware\RTE\Generation_Forecast\Wind")
PrevisionEolienne_2014 = pd.read_csv("PrevisionEolienne_2014.csv")
PrevisionEolienne_2015 = pd.read_csv("PrevisionEolienne_2015.csv")
PrevisionEolienne_2016 = pd.read_csv("PrevisionEolienne_2016.csv")
PrevisionEolienne_2017 = pd.read_csv("PrevisionEolienne_2017.csv")
PrevisionEolienne_2018 = pd.read_csv("PrevisionEolienne_2018.csv")
wind = pd.concat([PrevisionEolienne_2014, PrevisionEolienne_2015,
                PrevisionEolienne_2016, PrevisionEolienne_2017,
                PrevisionEolienne_2018])
del [PrevisionEolienne_2014, PrevisionEolienne_2015,
                PrevisionEolienne_2016, PrevisionEolienne_2017,
                PrevisionEolienne_2018]
wind['date_time'] = wind['Date'] + " " + wind['Heure']
wind.drop(['Date', 'Heure'], axis=1, inplace=True)
wind['date_time'] = pd.to_datetime(wind['date_time'])
wind.dtypes


os.chdir(r"C:\Users\yliu10\Desktop\Delaware\RTE\Consumption_load_Forecast\consumption")
prevision_conso2014 = pd.read_csv("prevision_conso2014.csv")
prevision_conso2015 = pd.read_csv("prevision_conso2015.csv")
prevision_conso2016 = pd.read_csv("prevision_conso2016.csv")
prevision_conso2017 = pd.read_csv("prevision_conso2017.csv")
prevision_conso2018 = pd.read_csv("prevision_conso2018.csv")
consumption = pd.concat([prevision_conso2014, prevision_conso2015,
                prevision_conso2016, prevision_conso2017,
                prevision_conso2018])
del [prevision_conso2014, prevision_conso2015,
                prevision_conso2016, prevision_conso2017,
                prevision_conso2018]
consumption = pd.melt(consumption, id_vars=['Jours/Heures'], var_name='Heures', value_name='consumption')
consumption['date_time'] = consumption['Jours/Heures'] + " " + consumption['Heures']
consumption.drop(['Jours/Heures', 'Heures'], axis=1, inplace=True)
consumption['date_time'] = pd.to_datetime(consumption['date_time'])
consumption = consumption.sort_values('date_time')
consumption.dtypes

os.chdir(r"C:\Users\yliu10\Desktop\Delaware\RTE\Consumption_load_Forecast\load")
conso_mix_RTE_2014 = pd.read_csv("conso_mix_RTE_2014.csv")
conso_mix_RTE_2015 = pd.read_csv("conso_mix_RTE_2015.csv")
conso_mix_RTE_2016 = pd.read_csv("conso_mix_RTE_2016.csv")
conso_mix_RTE_2017 = pd.read_csv("conso_mix_RTE_2017.csv")
conso_mix_RTE_2018 = pd.read_csv("conso_mix_RTE_2018.csv")