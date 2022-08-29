# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
#set working directory
os.chdir("/mydir")

#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#print(all_filenames)

#Looping to read all files and write(update) to csv
#Would not face ram limit due to read each file
#encoding='utf-8-sig' fix displaying issue
for f in all_filenames:
    #Read csv
    df = pd.read_csv(f)
    #write to csv
    df.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig', mode='a')
