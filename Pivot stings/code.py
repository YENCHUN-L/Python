# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:12:27 2022

@author: 
"""

import os
import pandas as pd
import numpy as np
import sys

# assign data of lists.  
data = {'產業': ['a','a','a','b', 'b', 'c', 'c'], '公司': ['a', 'b', 'c', 'd', 'e', 'f', 'g']}


# Create DataFrame  
df = pd.DataFrame(data)  

df['Value'] = 1


df1 = df.pivot_table(index='產業', values='Value', aggfunc='sum')

max_company = df1['Value'].max()

df2 = df.pivot_table(index='產業', columns='Value', values='公司', aggfunc=lambda x: ' '.join(x))

df2.columns = ['name']

df3= df2['name'].str.split(" ", n = max_company, expand = True)


