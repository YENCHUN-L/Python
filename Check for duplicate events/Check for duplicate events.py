# Check for duplicates events

# Change current working directory to desktop, file will read or write from or to desktop
import os
os.chdir(r"C:\Users\LocalUser\Desktop")

# Desire table
# event_id, account_id, event, created_date(to seconds data)

# Read library
import psycopg2
import numpy as np
import pandas as pd
import datetime as dt


df['created_date'] = pd.to_datetime(df['created_date'])

# get time difference when opp and status is the same
# https://stackoverflow.com/questions/48347497/pandas-groupby-diff
df['diff']=df.groupby(['account_id','event'])['created_date'].diff()

# change to seconds
df['diff'] = df['diff'].dt.total_seconds()

# time difference less than 3 sec
df2 = df[df['diff'] < 1]
