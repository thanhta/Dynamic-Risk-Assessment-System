"""
Author: Thanh Ta 
Date: March, 2024
Description: This script is used to ingest data from input_folder_path
"""

import pandas as pd
import glob
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(os.getcwd(), config['input_folder_path'])
output_folder_path = os.path.join(os.getcwd(), config['output_folder_path'])

#############Function for data ingestion
def merge_multiple_dataframe():
    """
    check for datasets, compile them together, and write to an output file
    Input: None
    Output: 
    Master dataset and list of ingested files stored into output_folder_path
    """
   
    logger.info(f"Starting merge_multiple_dataframe():")
    
    # master dataframe placeholder
    final_df = pd.DataFrame()
    
    logger.info(f"Retrieve files from: {input_folder_path}") 
    # compile datasets together and store ingested file names
    csv_files = glob.glob(f'{input_folder_path}/*.csv')
    logger.info(f"csv_files: {csv_files}")
    
    df = pd.concat(map(pd.read_csv, csv_files))
    final_df = df.reset_index(drop=True) 
    final_df.drop_duplicates(inplace=True)
    #logger.info(f"final_df: {final_df}")

    
    # write master dataset to an output file
    logger.info(f"Saving ingested dataframe to finaldata.csv")
    final_df.to_csv(os.path.join(output_folder_path,'finaldata.csv'), index=False)
    
     # write ingested filenames to an output file
    logger.info(f"Saving ingested file names to ingestedfiles.txt")
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as f:
        f.write(
            f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        for file in csv_files:
            f.write(file + "\n")


if __name__ == '__main__':
    logger.info("Execute ingestion.py")
    merge_multiple_dataframe()
