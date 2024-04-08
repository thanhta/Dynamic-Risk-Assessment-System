"""
Author: Thanh Ta 
Date: March, 2024
Description: This script is used to score a logistic regression model
in production against a test dataset
"""

import pandas as pd
import pickle
import os
from sklearn import metrics
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(os.getcwd(), config['test_data_path'])
output_model_path = os.path.join(os.getcwd(), config['output_model_path']) 

#################Function for model scoring
def score_model():
    """
    This function take a trained model, load test data, 
    and calculate an F1 score for the model relative to the test data
    Input: None
    Output: an F1 score for the model corresponding to the test data 
    which is written into latestscore.txt file
    """
    logger.info(f"Starting score_model")
    
    logger.info(f"Loading testdata.csv from {test_data_path}")
    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    
    logger.info(f"Loading trainedmodel.pkl from {output_model_path}")
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    model = pickle.load(open(model_path, "rb"))

    #features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    df = df.drop(columns=['corporation'])
    x_df = df.copy()
    y_df = x_df.pop("exited")
    predicted = model.predict(x_df)
    f1_score = metrics.f1_score(predicted, y_df)
    logger.info(f"From scoring.py --- f1_score: {f1_score}")
    
    with open(os.path.join(output_model_path, "latestscore.txt"), "w") as f:
        logger.info(
            f"F1 score =  {f1_score} store as latestscore.txt in {output_model_path}\n")
        f.write(f"F1 score =  {f1_score}")
    
    return f1_score
    
if __name__ == '__main__':
    logger.info("Invoking scoring.py")
    score_model()
