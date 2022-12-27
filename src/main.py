import json
import logging
import traceback
from config import Config
import tensorflow as tf

config = Config()
logging.basicConfig(level=logging.INFO)

model = config.load_model_objects()

# Spam / Ham prediction:


class Prediction:
    def __init__(self,  model: tf.keras.Model, configuration: Config = config) -> None:
        self.prediction_threshold = configuration.decision_threshold
        self.confidence = 0.0
        self.model = model

    def inference(self, text: str) -> dict:
        prediction = self.model.predict([text])
        self.confidence = prediction[0][0]
        return {"prediction": "spam", "confidence": str(self.confidence)} \
            if self.confidence > self.prediction_threshold \
            else {"prediction": "not-spam", "confidence": str(self.confidence)}


def handler(event, context):

    if event['rawPath'] == '/':
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "Service": "SMS Spam Classification API",
                "Status": "Active"
            })
        }

    elif event['rawPath'] == '/ping':
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "Service": "SMS Spam Classification API",
                "Status": "Active",
                "Ping": "Success"
            })
        }

    elif event['rawPath'] == '/predict':

        request_body = json.loads(event['body'])
        request_text = str(request_body['text'])

        prediction_pipeline = Prediction(model=model)

        try:
            result = prediction_pipeline.inference(text=request_text)

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "prediction": result.get("prediction"),
                    "confidence": result.get("confidence")
                })
            }
        except Exception as e:
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "Error": str(traceback.format_exc),
                    "Exception": str(e)
                })
            }
    else:
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "Service": "SMS Spam Classification API",
                "Status": "Active",
                "Message": "API method not allowed"
            })
        }
