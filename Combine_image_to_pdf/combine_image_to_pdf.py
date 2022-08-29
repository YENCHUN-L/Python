# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:15:00 2022

@author: Yen-Chun Liu
"""

#Merge all png to pdf
import img2pdf
import os
imgs = []
local = r'my/dir'
for fname in sorted(os.listdir(local), key=os.path.getmtime):
	if not fname.endswith(".png"):
		continue
	path = os.path.join(local, fname)
	if os.path.isdir(path):
		continue
	imgs.append(path)
with open("filename.pdf","wb") as f:
	f.write(img2pdf.convert(imgs))

#Remove all image file
#files_in_directory = os.listdir(local)
#filtered_files = [file for file in files_in_directory if file.endswith(".png")]
#for file in filtered_files:
#	path_to_file = os.path.join(local, file)
#	os.remove(path_to_file)