import os
from os import walk

#path
mypath = r"path"
os.chdir(mypath)

#get all folder in path
path_list = []
for root, dirs, files in walk(mypath):
  path_list.append(root)
path_list.pop(0)
del dirs, files, root
