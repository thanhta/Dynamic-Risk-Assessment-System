"""
Author: Thanh Ta 
Date: March, 2024
Description: 
This script is used to performs diagnostic tests 
related to your model as well as your data.
"""

import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import pickle

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])

logger.info(f"Retrieving testdata.csv from {test_data_path}")
test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

##################Function to get model predictions
def model_predictions(test_df):
    """
    This function read the deployed model and a test dataset, 
    to calculate predictions
    Input: None
    Output: A list containing all predictions
    """
    logger.info(f"Starting model_predictions")
    
    logger.info(f"Loading trainedmodel.pkl from {prod_deployment_path}")
    model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    model = pickle.load(open(model_path, "rb"))
    
    #features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    # Pick all rows starting from 1st column and 
    # exclude the last column (which is the target variable)
    x_test_df = test_df.iloc[:, 1:-1]
    predicted = model.predict(x_test_df)
    return predicted.tolist()

##################Function to get summary statistics
def dataframe_summary():
    """
    This function is  used to calculate summary statistics
    Input: None
    Output: A list containing all summary statistics
    """
    logger.info(f"Starting dataframe_summary")
    
    logger.info(f"Retrieving finaldata.csv from {dataset_csv_path}")
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    
    # Pick all rows starting from 1st column and exclude the last column 
    # (which is the target variable)
    X = df.iloc[:, 1:-1]
    stats = X.agg(["mean", "median", "std"]).to_dict(
        orient="index"
    )
    return stats
##################Function to get missing data
def missing_data():
    """
    This function is used to calculate % of missing data 
    on the dataset
    Input: None
    Output: A list containing % of missing data per column
    """

    logger.info(f"Starting missing_data")
    
    logger.info(f"Retrieving finaldata.csv from {dataset_csv_path}")
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    
    # compute missing data per column
    missing_data = df.isna().sum(axis=0)
    precent_missing_data = (missing_data / len(df) ) *100
    return precent_missing_data.tolist()

##################Function to get timings
def measure_ingestion_time():
    """
    This function is used to calculate timing of ingestion.py 
    Input: None
    Output: A list containing ingestion time statistics
    """
    logger.info(f"Starting measure_ingestion_time")
    
    # timing ingestion step
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - starttime
    return ingestion_time 
    
def measure_training_time():
    """
    This function is used to calculate timing of training.py 
    Input: None
    Output: A list containing ingestion time statistics
    """
    logger.info(f"Starting measure_training_time")
    
    # timing training step
    starttime = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - starttime
    return training_time 
         
def execution_time():
    """
    This function is used to calculate average timing of 
    ingestion.py and training.py 
    Input: None
    Output: A list containing summary statistics
    of average ingestion time and average training time
    """
    logger.info(f"Starting execution_time")
    time_index = 10
    
    logging.info(
        f"Calculating excute time for ingestion in a range of time: {time_index}")
    ingestion_time_list = []
    for i in range(time_index):
        ingestion_time = measure_ingestion_time()
        ingestion_time_list.append(ingestion_time)
        
    logging.info(
        f"Calculating excute time for training in a range of time: {time_index}")
    training_time_list = []
    for i in range(time_index):
        training_time = measure_training_time()
        training_time_list.append(training_time)    
    
    return [np.mean(ingestion_time_list), np.mean(training_time_list)]
  
##################Function to check dependencies
def outdated_packages_list():
    """
    This function is used to check the current and latest versions of 
    all the modules that your scripts use 
    Input: None
    Output: 
    A dataframe of "Package", "Installed Current Version", "Latest Version"
    """
    logger.info(f"Starting outdated_packages_list")
   
    # current version of dependencies
    with open('requirements.txt', 'r') as f:
        requirements = f.read().split('\n')
    requirements = [r.split('==') for r in requirements if r]
    requirements_df = pd.DataFrame(requirements, columns=["Package", "Requirement Version"])
    requirements_df["Package"] = requirements_df["Package"].str.lower()
    
    # Get outdated dependencies using PIP
    logger.info('Check outdated dependencies versions using PIP commands')
    os.system("pip list --outdated > out_dated_pip_list.txt")  
    outdated_df = pd.read_csv("out_dated_pip_list.txt", sep=r"\s+", skiprows=[1])
    outdated_df = outdated_df.drop(columns=['Type'])
    outdated_df["Package"] = outdated_df["Package"].str.lower()
    outdated_df.columns = ["Package", "Installed Version","Latest Version"]
    
    # Merge requirements_df and outdated_df
    packages_df = requirements_df.merge(outdated_df, how="left", on="Package")
    
    # If we're already using the latest version of a module, we fill latest with the requirements version:
    packages_df['Latest Version'].fillna(packages_df['Requirement Version'], inplace=True)
    
    packages_df.to_csv("package_version.csv", index=False)
    return packages_df

if __name__ == "__main__":
    logger.info("Invoking diagnostics.py")
    predictions = model_predictions(test_df)
    logger.info(f'predictions =  {predictions}')
    stats = dataframe_summary()
    logger.info(f'stats =  {stats}')
    precent_missing_data = missing_data()
    logger.info(f'precent_missing_data =  {precent_missing_data}')
    timings = execution_time()
    logger.info(f'timings=  {timings}')
    packages_df = outdated_packages_list()
    logger.info(f'packages_df: \n {packages_df}')
