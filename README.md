# Spam-Classifier V2

Downloaded the dataset "sms spam classifier" from kaggle and extended the dataset using spam sms from my personal mobile phone, Thus making it a personalized dataset. This also helped to handle the class imbalance problem present in the initial kaggle dataset. Analysed the text using text cloud. Performed standard text processing techniques. Split the Dataset into 70-30 proportion. Trained an LSTM model for 10 epochs and achieved a validation accuracy and loss of 0.99 and 0.03 respectively. The model is served via Flask and has been containerized via Docker.

## Running the project:

#### Using Flask:

Make sure the virtual environment: "env" is active

from the root folder, enter cmd:
    `python main.py`

Then Send POST request to URL: http://0.0.0.0:8000/predict to get the prediction


#### Using Docker:

Make sure the virtual environment: "env" is active

Build the docker image using:
    `docker build -t <your_container_name>` container name example: spamclassifier_api

Run the build image:
    `docker run -p 8000:8000 -t <your_container_name>` container name example: spamclassifier_api

Note: Port 8000 is exposed for Docker container and Flask app is configured to run on port 8000

Then Send POST request to URL: http://{default docker ip}:8000/predict to get the prediction

