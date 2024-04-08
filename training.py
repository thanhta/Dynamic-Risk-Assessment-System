"""
Author: Thanh Ta 
Date: March, 2024
Description: This script is used to train a logistic regression model
"""

import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path']) 
output_model_path = os.path.join(os.getcwd(), config['output_model_path']) 

#################Function for training the model
def train_model():
    """
    Trained a LogisticRegression model
    Input: None
    Output: A trained LogisticRegression model and stored it into model_path
    """
    
    #use this logistic regression for training
    logger.info(f'Starting to train a Logistic regression model')
    
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    logger.info(f'Reading ingested data from finaldata.csv of {dataset_csv_path}')
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    
    #features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    df = df.drop(columns=['corporation'])
    x_df = df.copy()
    y_df = x_df.pop("exited")
    
    #fit the logistic regression to your data
    logger.info(f'fit the logistic regression to the ingested data')
    #logger.info(f"x_df: {x_df}\n")
    #logger.info(f"y_df: {y_df}\n")
    
    model.fit(x_df, y_df)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    logger.info(f'write the trained model to a file called: trainedmodel.pkl of {output_model_path}')
    pickle.dump(model, open(os.path.join(output_model_path, "trainedmodel.pkl"), "wb"))
    

if __name__ == "__main__":
    logger.info(f'Executing training.py:')
    train_model()    