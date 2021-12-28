import uvicorn
import logging
import traceback
from pydantic import BaseModel
from config import Config
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences

config = Config()
app = FastAPI()
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
        txt = pad_sequences(txt, maxlen=self.maxlen)
        prediction = self.model.predict(txt)
        self.confidence = prediction[0][0]
        return "spam", self.confidence if self.confidence > self.prediction_threshold \
            else "ham", self.confidence


class PredictionRequest(BaseModel):
    text: str


prediction_pipeline = Prediction(model=model, tokenizer=tokenizer)


@app.get("/ping")
async def ping():
    return 'server running'


@app.get("/")
async def home():
    return 'Spam Classification API'


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        prediction, confidence = prediction_pipeline.inference(text=request.text)
        return JSONResponse(content=jsonable_encoder({'prediction': prediction, 'confidence': confidence}))
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({'Error': traceback.format_exc(), 'Exception': e}))


if __name__ == '__main__':
    uvicorn.run(app, host=config.host_address, port=config.port)
