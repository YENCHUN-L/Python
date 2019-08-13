# Data info script
data = df.copy() # dataframe

# Get info
from io import StringIO
import pandas as pd
def process_content_info(content: pd.DataFrame):
    content_info = StringIO()
    content.info(buf=content_info)
    str_ = content_info.getvalue()

    lines = str_.split("\n")
    table = StringIO("\n".join(lines[3:-3]))
    datatypes = pd.read_table(table, delim_whitespace=True, 
                   names=["column", "count", "null", "dtype"])
    datatypes.set_index("column", inplace=True)

    info = "\n".join(lines[0:2] + lines[-2:-1])

    return info, datatypes

info_table = process_content_info(content=data)
info_table = info_table[1]
info_table['index'] = info_table.index
del  info_table['count']

# Get describe
des = data.describe().T
des['index'] = des.index


# Get duplicate column
index = list(data)
y = list()
for i in range(0,len(data),1):
    x = data.iloc[:,i].duplicated()
    if x.any() == True:
        t = True
        y.append(t)
    else:
        f = False
        y.append(f)
dup = pd.DataFrame()
dup['index'] = index
dup['dup'] = y

data_information = pd.merge(info_table, des, how='left', left_on='index', right_on='index')
data_information = pd.merge(data_information, dup, how='left', left_on='index', right_on='index')

del f, i, t, index, y, data, x, dup, des, info_table

