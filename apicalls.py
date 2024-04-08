"""
Author: Thanh Ta 
Date: March, 2024
Description: This script is used to invoke Flask APIs
"""

import requests
import json
import os
import logging

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# API endpoints
#default_endpoint = f"{URL}"
prediction_endpoint = f"{URL}/prediction"
scoring_endpoint = f"{URL}/scoring"
summarystats_endpoint = f"{URL}/summarystats"
diagnostics_endpoint = f"{URL}/diagnostics"

#Call each API endpoint and store the responses
prediction_response = requests.post(
    url=prediction_endpoint,
    headers={"Content-Type": "application/json"},
    json={'filepath': "testdata/testdata.csv"}
).json()
logger.info(f"prediction_response: {prediction_response}\n")

scoring_response = requests.get(scoring_endpoint).json()
logger.info(f"scoring_response: {scoring_response}\n")

summarystats_response = requests.get(summarystats_endpoint).json()
logger.info(f"summarystats_response: {summarystats_response}\n")

diagnostics_response = requests.get(diagnostics_endpoint).json()
logger.info(f"diagnostics_response: {diagnostics_response}\n")

#responses = #combine reponses here
#combine all API responses
responses = {
    'Predictions': prediction_response,
    'F1 Scoring': scoring_response,
    'Summary Stats': summarystats_response,
    'Diagnostics Results': diagnostics_response
}

with open('config.json','r') as f:
    config = json.load(f)
   
#write the responses to your workspace
output_model_path = os.path.join(os.getcwd(), config['output_model_path']) 
with open(os.path.join(output_model_path, "apireturns2.txt"), "w") as file:
    json.dump(responses, file)

