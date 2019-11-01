# Read library
import psycopg2
import numpy as np
import pandas as pd

# Connetion info
con = psycopg2.connect(
            host = "xxx",
            port = "xxx",
            database = "xxx",
            user = "xxx",
            password = "xxx",
            )

# Connect to database
cur = con.cursor()

# Query to execute
cur.execute("SQL query;")
              
colnames = [desc[0] for desc in cur.description]


# Read query to np.array
df = pd.DataFrame(np.array(cur.fetchall()))

# Name columns
df.columns = colnames

del colnames
# Connetion close
cur.close() 

# Clear connection info
con.close()


