import json
import logging
import traceback
from config import Config
from tensorflow.keras.preprocessing.sequence import pad_sequences

config = Config()
logging.basicConfig(level=logging.INFO)

model, tokenizer = config.load_model_objects()

# Spam / Ham prediction:


class Prediction:
    def __init__(self,  model, tokenizer, configuration:Config = config) -> None:
        self.maxlen = configuration.max_length
        self.prediction_threshold = configuration.decision_threshold
        self.confidence = 0.0
        self.tokenizer = tokenizer
        self.model = model

    def inference(self, text: str) -> tuple:
        txt = self.tokenizer.texts_to_sequences([text])
        txt = pad_sequences(txt, maxlen=int(self.maxlen))
        prediction = self.model.predict(txt)
        self.confidence = prediction[0][0]
        return ("spam", self.confidence) if self.confidence > self.prediction_threshold \
            else ("ham", self.confidence)


def handler(event, context):

    if event['rawPath'] == '/' or event['rawPath'] == '/ping':
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

    elif event['rawPath'] == '/predict':

        request_body = json.loads(event['body'])
        request_text = str(request_body['text'])

        prediction_pipeline = Prediction(model=model, tokenizer=tokenizer)

        try:
            prediction, confidence = prediction_pipeline.inference(text=request_text)

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "prediction": prediction,
                    "confidence": str(confidence)
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
