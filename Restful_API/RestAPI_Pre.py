# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:58:33 2019

@author: yliu10
"""

## Import packages
from flask import Flask, request, jsonify
import pandas

## Initialize app
app = Flask(__name__)

## Read in predictions.csv and movies.csv using Pandas
#predictions = pandas.read_csv("Data/predictions.csv").dropna()

#movies = pandas.read_csv("Data/movies.csv").dropna()
## Combine both files to allow to return complete information about the movie
predictions = pandas.read_csv(r"C:\Users\yliu10\Desktop\RMT\datamart.csv").sort_values(['userid', 'prediction'], ascending=[True, False])

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
    if "userid" in content and type(content["userid"]) == int:
        userid = content["userid"]
    else:
        return "'userid' is required and should be an Integer."
        sys.exit("'userid' is required and should be an Integer.")
        
    if "count" in content and type(content["count"]) == int:
        count = content["count"]
    else:
        count = 5
    
    # filter predictions for the given userId
    predict = predictions[predictions.userid == userid].head(count)
    
    # select movieId, title and prediction and transform to list
    top_ratings = list(predict[["userid", "artistid", "prediction"]].T.to_dict().values())
    
    # Return the result to the API
    return jsonify(top_ratings)

### Put endpoint online
if __name__ == '__main__':
    app.run(host='localhost', port=6000)