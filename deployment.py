"""
Author: Thanh Ta 
Date: March, 2024
Description: 
This script is used to deploy the latest pickle file, 
the latestscore.txt value and 
the ingestfiles.txt file into the deployment directory
"""


import pandas as pd
import os
import json
import logging
import shutil

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path']) 
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])
output_model_path = os.path.join(os.getcwd(), config['output_model_path']) 

####################function for deployment
def store_model_into_pickle():
    """
    This function copy the pickle file, the latestscore.txt value, 
    and the ingestfiles.txt file into the deployment directory
    Input: None
    Output: 
    copy the pickle file, the latestscore.txt value, 
    and the ingestfiles.txt file into the deployment directory
    """
    logger.info(f"Starting store_model_into_pickle")
    
    logger.info(f"copy the ingestfiles.txt file into {prod_deployment_path}")
    shutil.copy(os.path.join(dataset_csv_path,'ingestedfiles.txt'),
                prod_deployment_path)
                
    logger.info(f"copy the pickle file into {prod_deployment_path}")
    shutil.copy(os.path.join(output_model_path,'trainedmodel.pkl'),
                prod_deployment_path)
    
    logger.info(f"copy the latestscore.txt file into {prod_deployment_path}")
    shutil.copy(os.path.join(output_model_path,'latestscore.txt'),
                prod_deployment_path)
    
if __name__ == "__main__":
    logger.info("Invoking deployment.py")
    store_model_into_pickle()    

