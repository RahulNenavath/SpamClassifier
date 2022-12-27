# Spam-Classifier

## Description
- New Updates:
  - Migrated from TF v2.6 to TF v2.11
  - Made use of TF Datasets to load and process data
  - Integrated tokenizer and model into a single end-to-end model
- Extended Dataset: Downloaded the dataset "sms spam classifier" from kaggle and extended the dataset using spam sms from my personal mobile phone, Thus making it a personalized dataset. 
- Analysed the text using text cloud. 
- Standardization: Lower text and removing punctuations & special chars. 
- Data split: Train 70% & Test 30%. 
- Models Trained: Conv-1D model & LSTM model
- Epochs Trained: 10 epochs
- K-Folds Trained: 10 folds
- Metrics Computed: Precision | Recall | F1-Score | ROC Curve
- Model Selection: LSTM model has an overall higher mean F-1 score than Conv model for 10-Fold CV. However, on looking at the individual F-1 score of spam & not-spam classes, Conv model gives a better score. <br>
Check the Notebook for detailed evaluation metrics. The module is containerized via Docker
- Model Enhancements: Pruning was done to add sparsity in the model to reduce the size of the model without losing on accuracy

## Input and Output:
Input: { "text": string } <br>
Output: { "prediction": "spam/not-spam", "confidence": range (0 to 1) }

## Tech Stack:
Python 3.8, Tensorflow 2.11, Pandas, Numpy, Docker, AWS Lambda, AWS ECR, and GitHub CI/CD.

## Deployment:
This project has been deployed as a AWS Lambda function using container image from AWS ECR service and made available as an API. AWS Lambda functions are cost-effective solutions for personal projects that have minimal network traffic. <br> In order to facilitate for a higher network traffic, it can easily be done so by pulling the ECR image into an EC2 container and deploying the container.