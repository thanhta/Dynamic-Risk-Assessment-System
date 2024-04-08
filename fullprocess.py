"""
Author: Thanh Ta 
Date: March, 2024
Description: 
This script is used to automate ML model scoring, monitoring, 
and re-deployment process.
"""
import os
import json
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import subprocess
import logging
import pandas as pd
from sklearn import metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])
input_folder_path = os.path.join(os.getcwd(), config['input_folder_path']) 
output_folder_path = os.path.join(os.getcwd(), config['output_folder_path']) 
output_model_path = os.path.join(os.getcwd(), config['output_model_path'])

##################Check and read new data
def check_new_data():
    """
    check if there are new ingested data files
    Input: None
    Output:
    Return True if there are new ingested data files
    Return False if there are NO new ingested data files
    """
    logger.info(f"Check if any new ingested file")
    
    #first, read ingestedfiles.txt
    ingested_filepath = os.path.join(prod_deployment_path,'ingestedfiles.txt')
    # Skip the fisrt line because it is recorded the timestamp 
    # where the files are previously ingested
    with open(ingested_filepath) as f:
        ingested_files = {line.strip('\n') for line in f.readlines()[1:]}
    logger.info(
        f"ingested files from ingestedfiles.txt: {ingested_files}\
          from ingested_filepath: {ingested_filepath}")

    #second, determine whether the source data folder has files that 
    #aren't listed in ingestedfiles.txt
    input_source_files = set(os.listdir(input_folder_path))
    logger.info(f"input_source_files: {input_source_files}")
    if len(input_source_files.difference(ingested_files)) == 0:
        logging.info("No new data found!")
        return False
    else:
        logging.info("Found new data !")
        return True

def check_model_drift():
    """
    check if the model has drifted or not 
    Input: None
    Output:
    Return True if the model has drifted
    Return False if the model has NOT drifted
    """
    #check whether the score from the deployed model is different from the score 
    #from the model that uses the newest ingested data
    logger.info(f"Checking for model drift")
    latest_score_filepath = os.path.join(prod_deployment_path,'latestscore.txt')
    with open(latest_score_filepath) as f:
        latest_score = float(f.readline().split("=")[1].strip())
    logger.info(f"latest_score:  {latest_score}")

    output_folder_filepath = os.path.join(output_folder_path, 'finaldata.csv')
    new_data_df = pd.read_csv(output_folder_filepath)
    y_pred = diagnostics.model_predictions(new_data_df)
    y_df = new_data_df["exited"]
    new_score = metrics.f1_score(y_pred, y_df)
    logger.info(f"new_score:  {new_score}")

    if(new_score >= latest_score):
        logger.info("No model drift occurred !")
        return False
    else:
        logger.info("Model drift occurred !")
        return True
##################Check for new data
# check for new data, 
# if no new data, the process end here
# if new data exists, combine all of the data files
# into one dataset
if (check_new_data() == False):
    exit()
    
logger.info(f"Ingest new data files into one dataset")
ingestion.merge_multiple_dataframe()

##################Checking for model drift
# if no model drift, the process end here
# if the model has drifted,
# Deploy the best model
# Get Diagnostics and reports

if (check_model_drift() == False):
    exit()

##################Re-deployment
#Now, model has drifted, we need:
# Re-training
# Calulate new score
# Deploy the new model
logger.info("Model drift has occurred !!!")

logger.info("Re-training model")
training.train_model()

logger.info("Re-scoring model")
scoring.score_model()

logger.info("Re-deploying model")
deployment.store_model_into_pickle()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
logger.info("Invoking score_model to generate and store confusionmatrix2.png")
reporting.score_model()

logger.info("Invoking apicalls.py")
os.system("python apicalls.py")


