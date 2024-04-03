# A Dynamic Risk Assessment System

- Project **A Dynamic Risk Assessment System** is the fourth project from the Udacity's ML DevOps Engineer Nanodegree 

## Project Background
Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model you create and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the end of your work, though. Your industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, you need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

## Project Description
The objective of this project is to implement an MLOps Life cycle for an ML Model to predict which customers that are most likely to exit their contracts. This project will check in regular intervals for new datasets and acts upon any new data to see if the model is needed to be-trained. The ML model is evaluated for model drift, re-train the ML model when the model is drifted. Then, it will generate a new Risk Assessment report which captures the model performance, data quality and timing of execution and the report is saved and updated in a production deployment folder. Model Drift is defined as the situation when a model tends to perform worse over time when tested on new data sets.

This project follows the following steps:
### Data ingestion
Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.
### Training, scoring, and deploying
Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
### Diagnostics
Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
### Reporting
Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.
### Process Automation
Create a script and cron job that automatically run all previous steps at regular intervals.

## Files and data description
The files and directories in the root directory are organized as the following:
* Project files
  * **ingestion.py**: This script is used to ingest data from input folder and covert them to pandas dataframes
  * **training.py**: This script is used to train a logistic regression model
  * **scoring.py**: This script is used to score a logistic regression model against a test dataset
  * **reporting.py**: This script is used to to generates plots related to the ML model's performance such as confusion matrix
  * **deployment.py**: This script is used to to deploy the latest model pickle file, the latestscore value and etc to production
  * **diagnostics.py**: This script is used to generate summary statistics of the input data, quality of the input data, ingestion and training execution timings as well as the current and latest versions of packages used in this project
  * **app.py**: This script is used to implement the Flash API end points to infer the prediction output, to get model performance and to collect various summary statistics 
  * **apicalls.py**: This script is used to call all of the Flash API end points and generate a consolidated report
  * **fullprocess.py**: This script is used to monitor for new data availability, to evaluate the model drift, to retrain and redeploy an updated ML model if model drift is detected.

* Other files
  * **requirements.txt**: This text file is defined the current versions of all of the dependent python modules used in this project
  * **config.json**: This json file is used to define the physical and the logical mapping of vruous data folders
  * **cronjob.txt** A crontab file that runs the fullprocess.py script one time every 10 min.

* Data Folders
  * **practicedata** : stores the data files which are used to test the data ingestion script
  * **sourceddata** : stores the data files which are used to run the data ingestion script in production
  * **ingesteddata** : stores the output csv file for the dataframes generated after the data ingestion process and record the input file names used during the data ingestion process
  * **testdata**: stores the test data file used to evaluate the model performance
  * **practicemodels** stores the pickle model file, confusion matrix plot, performance score, consolidated report generated from the output of the Flash API end points during practice
  * **production_deployment** stores the pickle model file, performance score, and record the input file names used during the data ingestion process
  * **models** stores the pickle model file, confusion matrix plot, performance score, consolidated report generated from the output of the Flash API end points during new data ingestion which requires re-training the model





