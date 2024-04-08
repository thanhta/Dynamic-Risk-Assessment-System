"""
Author: Thanh Ta 
Date: March, 2024
Description: This script is used to set up Flask APIs
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import pickle
import json
import os
import logging
from diagnostics import (
    missing_data,
    dataframe_summary,
    execution_time,
    model_predictions,
    outdated_packages_list,
)
from scoring import score_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

######################Set up variables for use in our script
app = Flask(__name__)

app.config.from_pyfile('settings.py')

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

#######################Default Endpoint
@app.route('/')
def index():
    return "Welcome Risk Assessment APIs"

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    logger.info(f"Invoking predict()")
    
    #filepath = request.json.get['filepath']
    filepath = request.get_json()['filepath']

    df = pd.read_csv(filepath)
    prediction_list = model_predictions(df)
    
    #return a list of prediction outputs
    return jsonify({"predictions": prediction_list})

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    logger.info(f"Invoking score()")
    f1_score = score_model()
    logger.info(f"From API --- f1_score: {f1_score}")
    
    #F1 score number
    return jsonify({"f1_score": f1_score})
 
#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    logger.info(f"Invoking stats()")    
    stats_list = dataframe_summary()
    
    #return a list of all calculated summary statistics
    return jsonify(stats_list)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    logger.info(f"Invoking diagnostics()")
    # Check precent missing data
    precent_missing_data = missing_data()
    # Check execution timings
    durations = execution_time()
    # Check outdated_packages_list
    dependancies_check = outdated_packages_list().to_dict()
    #return value for all diagnostics
    diagnostics = {
            "missing_data_percentage": precent_missing_data,
            "execution_durations": durations,
            "outdated_dependencies": dependancies_check ,
        }
    
    return jsonify(diagnostics)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
