# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:02:10 2021

@author: ycl
"""

import pandas as pd
import numpy as np
import sys
import glob
import os
import re, string
from selenium import webdriver
from selenium.common import exceptions as ex
import time as time
import datetime as datetime
import math
import keyboard
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from csv import reader
from bs4 import BeautifulSoup as bs
from selenium.common.exceptions import TimeoutException
import mouse
import io

local = os.path.expanduser(r"~/Downloads")
os.chdir(local)
chromedriver = ('chromedriver.exe')
#https://chromedriver.chromium.org/downloads
driver = webdriver.Chrome(executable_path=chromedriver)
driver.maximize_window()
driver.get("https://mops.twse.com.tw/mops/web/index")

#Input for website
stockcode = str(6670)
year = 111
current_month = 7

month = current_month - 1

#string for file name
year_for_input = str(year)
month_for_input = str(month+1).zfill(2)


driver.find_element_by_xpath('//*[@id="keyword"]').send_keys(stockcode)

element = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="rulesubmit"]')))

element.click()

element = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="mm7"]')))

element.click()

element = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="mm7"]/ul/li[10]')))

element.click()

driver.find_element_by_xpath('//*[@id="co_id"]').send_keys(stockcode)

driver.find_element_by_xpath('//*[@id="year"]').send_keys(year_for_input)

element = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="month"]/option['+month_for_input+']')))

element.click()

time.sleep(2)

#driver.find_element_by_xpath('//*[@id="co_id"]').send_keys(stockcode)

mouse.get_position()

mouse.move(1414, 352, absolute=True, duration=0.2)

mouse.click('left')

time.sleep(2)

mouse.wheel(-5)

time.sleep(2)

#Screen shot for first page
driver.save_screenshot('main.png')


#Get list for all company
nodes = driver.find_elements_by_xpath('//*[@id="t15sf_fm"]/table')
for node in nodes:
    print(node.text)
lst_code_str = node.get_attribute('innerText')

df = pd.read_csv(io.StringIO(lst_code_str), sep='\t')

df.info()

lst = df['母/子公司代號']

len(lst)

time.sleep(2)


#Go through all company screen shot
for i, j in zip(range(2, len(lst)+2), lst): 
    element = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="t15sf_fm"]/table/tbody/tr[' + str(i) + ']/td[3]/input')))
    
    element.click()
    
    time.sleep(5)
    
    import pyscreenshot as ImageGrab
    
    # Screen shot range
    im = ImageGrab.grab(bbox=(12,   # X1
                            148,   # Y1
                            800,   # X2
                            600))  # Y2
    
    # save as image
    im.save(str(j) + ".png")
    
    window_after = driver.window_handles[1]
    driver.switch_to.window(window_after)
    driver.close()
    window_before = driver.window_handles[0]
    driver.switch_to.window(window_before)
    #driver.close()
    #driver.save_screenshot("screenshot.png")
driver.quit()

#Merge all img to pdf
import img2pdf
imgs = []
for fname in sorted(os.listdir(local), key=os.path.getmtime):
	if not fname.endswith(".png"):
		continue
	path = os.path.join(local, fname)
	if os.path.isdir(path):
		continue
	imgs.append(path)
with open("公告_" + year_for_input + month_for_input +".pdf","wb") as f:
	f.write(img2pdf.convert(imgs))

#Remove all image file
files_in_directory = os.listdir(local)
filtered_files = [file for file in files_in_directory if file.endswith(".png")]
for file in filtered_files:
	path_to_file = os.path.join(local, file)
	os.remove(path_to_file)