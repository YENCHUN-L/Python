from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import time
import pandas as pd
import pickle as pk
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait 
import urllib
import urllib.parse
import json
import time
import requests
import pandas as pd
import logging
from datetime import datetime
import calendar
from datetime import date, timedelta

Date = str()
Time = str()
Temperature = str()
Dew = str()
Humidity = str()
Wind = str()
Speed = str()
Gust = str()
Pressure = str()
Precip = str()

    
CHROMEDRIVER_PATH = "/chromedriver.exe"

options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(chrome_options=options, executable_path=CHROMEDRIVER_PATH)

d1 = date(2013, 12, 31)  # start date
d2 = date(2018, 12, 31)  # end date
delta = d2 - d1         # timedelta
date_list = []
for d in range(delta.days + 1):
    date_list.append(d1 + timedelta(d))

City = "lille"
start_time = datetime.now()
all_date_df = pd.DataFrame()
for r in range(len(date_list)):
    print(r)
    Date = str(date_list[r])
#    Date = '2019-2-13'
    url ='https://www.wunderground.com/history/daily/fr/'+City+'/LFQQ/date/'+Date+'?cm_ven=localwx_history'
    
#    url ='https://www.wunderground.com/history/daily/fr/lille/LFQQ/date/2019-2-13?cm_ven=localwx_history'
    driver.get(url)
    data  = WebDriverWait(driver, 30).until(lambda d: d.find_element_by_id('history-observation-table')).text
    data_split = data.split("\n")
    #col_names = data_split[0].split()
    #del col_names[8]
    #del col_names[6]
    #del col_names[3]
    #print(col_names)
    
    weather_data = data_split[1:]
    today_data = []
    rec = []
    for i in range(len(weather_data)):
        if(i%4==0): 
            Time = weather_data[i]
        try:    
            if(i%4==1): 
                Temperature = weather_data[i].split()[0]
        except IndexError:
            continue
        try:    
            if(i%4==1):    
                Dew = weather_data[i].split()[2]
        except IndexError:
            continue              
        try:    
            if(i%4==1):    
                Humidity = weather_data[i].split()[4]
        except IndexError:
            continue
        try:    
            if(i%4==2): 
                Wind = weather_data[i]
        except IndexError:
            continue
        try:    
            if(i%4==3): 
                Speed = weather_data[i].split()[0]
        except IndexError:
            continue
        try: 
            if(i%4==3): 
                Gust = weather_data[i].split()[2]
        except IndexError:
            continue
        try: 
            if(i%4==3): 
                Pressure = weather_data[i].split()[4]
        except IndexError:
            continue
        try: 
             if(i%4==3): 
                Precip = weather_data[i].split()[6]
        except IndexError:
            continue
        if(i%4==0 and len(Precip)>0):
            rec = [Date, Time, Temperature, Dew, Humidity, Wind, Speed, Gust, Pressure, Precip]
            today_data.append(rec)
            Precip = ""
    if(len(Wind)>4):
        continue
    else:
        today_df = pd.DataFrame(today_data)
        all_date_df=pd.concat([all_date_df, today_df])
   
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    
all_date_df.to_csv("WeatherLille.csv")

#VIRIYAKOVITHYA Ekapope
#Yen Chun Liu
