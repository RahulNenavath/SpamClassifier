import os
from joblib import load
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from configparser import ConfigParser
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={
    r'/*':{
        'origins':'*'
    }
})

config = ConfigParser()
config.read('./config/config.ini')

# Set the Dir paths
root_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(root_dir,config['DIR']['obj_dir'])

# Set the Const paths
model_path = os.path.join(model_dir,config['objects']['model'])
tokenizer_path = os.path.join(model_dir,config['objects']['tokenizer'])
maxlen = int(config['model']['maxlen'])

# Load the serialized pickel and tensorflow objects 
class LoadPklObject:
    def __init__(self, model_path, tokenizer_path):

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def load_object(self):

        model = load_model(self.model_path)
        tokenizer = load(open(self.tokenizer_path, "rb"))

        return model, tokenizer

load_pkl_file = LoadPklObject(model_path, tokenizer_path)

model, tokenizer = load_pkl_file.load_object()

# Spam / Ham prediction:
class Prediction:
    def __init__(self, sms, max_len = maxlen):
        self.sms = sms
        self.maxlen = max_len
        self.prediction_threshold = 0.5
        self.class_lables = {0.0:"ham", 1.0:"spam"}

    def preprocess(self):
        sms_seq = tokenizer.texts_to_sequences([self.sms])
        self.sms_seq = pad_sequences(sms_seq, maxlen=self.maxlen)
    
    def getScore(self):
        self.confidence = model.predict(self.sms_seq)[0][0]
    
    def getDecision(self):
        if self.confidence > self.prediction_threshold:
            self.output = float(1)
        else:
            self.output = float(0)

        self.predicted_class = self.class_lables[self.output]

    def run(self):
        self.preprocess()
        self.getScore()
        self.getDecision()
        return self.predicted_class, str(self.confidence) 

@app.route("/test", methods=["GET"])
def test():
    return jsonify({'response':'working'})


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":

        json_ = request.json

        sms = json_["sms"]

        prediction_obj = Prediction(sms)
        predicted_class, confidence = prediction_obj.run()
    
        return jsonify({"prediction": predicted_class , "confidence":confidence})


if __name__ == "__main__":
    app.run(debug=True, host=config['server']['host'], port=config['server']['port'])
