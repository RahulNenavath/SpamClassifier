import os
import logging
from joblib import load
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()
logging.basicConfig(level=logging.INFO)


class Config:
    def __init__(self):
        # Project Config
        self.project_folder = os.path.dirname(__file__)
        self.model_folder_path = os.path.join(self.project_folder, 'model')
        # Model Config
        self.model_name = os.getenv('MODEL_NAME')
        self.tokenizer_name = os.getenv('TOKENIZER_NAME')
        self.model_path = os.path.join(self.model_folder_path, self.model_name)
        self.tokenizer_path = os.path.join(self.model_folder_path, self.tokenizer_name)
        self.max_length = os.getenv('MAX_LEN')
        self.decision_threshold = 0.75
        # Deployment Config
        self.host_address = '0.0.0.0'
        self.port = os.getenv('PORT')

        logging.info(f'Configuration Loaded Successfully!')

    def load_model_objects(self):
        model = load_model(self.model_path)
        tokenizer = load(open(self.tokenizer_path, "rb"))
        logging.info(f'Model and Tokenizer Loaded Successfully!')
        return model, tokenizer
