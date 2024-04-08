"""
Author: Thanh Ta 
Date: March, 2024
Description: This script is used to generates plots 
related to the ML model's performance.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt
import json
import os
import logging

from diagnostics import model_predictions

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path'])
output_model_path = os.path.join(os.getcwd(), config['output_model_path']) 

##############Function for reporting
def score_model():
    """
    This function is used to calculate a confusion matrix using 
    the test data and the deployed model
    and save confusion matrix plot to a file 
    
    Input: None
    Output: save confusion matrix plot to a file 
    """
    logger.info(f"Starting score_model")
    
    logger.info(f"Retrieving testdata.csv from {test_data_path}")
    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    y_pred = model_predictions(test_df)
    y_true = test_df['exited']

    # calculate confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    
    # Save confusion matrix plot to a file
    logger.info(f"confusionmatrix2.png is stored in {output_model_path}")
    plt.savefig(os.path.join(output_model_path, "confusionmatrix2.png") )
    

if __name__ == '__main__':
    score_model()
