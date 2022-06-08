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
data = {'industry': ['a','a','a','b', 'b', 'c', 'c'], 'companies': ['a', 'b', 'c', 'd', 'e', 'f', 'g']}


# Create DataFrame  
df = pd.DataFrame(data)  

df['Value'] = 1


df1 = df.pivot_table(index='industry', values='Value', aggfunc='sum')

max_company = df1['Value'].max()

df2 = df.pivot_table(index='industry', columns='Value', values='companies', aggfunc=lambda x: ' '.join(x))

df2.columns = ['name']

df3= df2['name'].str.split(" ", n = max_company, expand = True)


