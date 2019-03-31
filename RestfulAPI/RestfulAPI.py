# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:28:46 2019

@author: eviriyakovithya
"""

import os
os.chdir(r"C:\Users\yliu10\Desktop")


## Import packages
from flask import Flask, request, jsonify
import pandas

## Initialize app
app = Flask(__name__)

## Read in predictions.csv and movies.csv using Pandas
predictions = pandas.read_csv("datamart.csv").dropna()
artists = pandas.read_csv("artists.csv").dropna()
# select only id and name column
artists = artists[['ArtistID', 'name']]
artists['ArtistID']=artists['ArtistID'].astype(int)

## Combine both files to allow to return complete information about the movie
predictions = predictions.merge(artists, on="ArtistID", how="left").sort_values(['userId', 'prediction'], ascending=[True, False])

## How this is done in Spark (information)
#movies = spark.read.option("header", "true").csv("movies.csv")\
#                    .select("movieId", "title")\
#                    .repartition("movieId")
            
#predictions = predictions.join(movies, "movieId", "left")\
#                    .orderBy(col("userId"), col("prediction").desc())\
#                    .cache()

## The application definiton
### Endpoint - one route /ratings/top - one HTTP verb = POST
@app.route("/ratings/top", methods=["POST"])
def top_ratings():
    ## read the body of the API call
    content = request.get_json()
    
    ## Interpretation of body
    if "userId" in content and type(content["userId"]) == int:
        userId = content["userId"]
    else:
        return "'userId' is required and should be an Integer."
        sys.exit("'userId' is required and should be an Integer.")
        
    if "count" in content and type(content["count"]) == int:
        count = content["count"]
    else:
        count = 5
    
    # filter predictions for the given userId
    predict = predictions[predictions.userId == userId].head(count)
    
    # select movieId, title and prediction and transform to list
    top_ratings = list(predict[["ArtistID", "name", "prediction"]].T.to_dict().values())
    
    # Return the result to the API
    return jsonify(top_ratings)

### Put endpoint online
if __name__ == '__main__':
    app.run(host='localhost', port=6000)
