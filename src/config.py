import os
import re
import string
import logging
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)


class Config:
    def __init__(self):
        # Project Config
        self.project_folder = os.path.dirname(__file__)
        self.model_folder_path = os.path.join(self.project_folder, 'model')
        self.model_path = os.path.join(self.model_folder_path, 'pruned_spam_detector')
        # Model Config
        self.decision_threshold = 0.75
        # Deployment Config
        self.host_address = '0.0.0.0'
        self.port = os.getenv('PORT')
        logging.info(f'TF Version: {tf.__version__}')
        logging.info(f'Configuration Loaded Successfully!')

    def load_model_objects(self):

        @tf.keras.utils.register_keras_serializable(package='Custom', name='custom_standardization')
        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            remove_special_chars = tf.strings.regex_replace(lowercase, r"[^a-zA-Z0-9 ]", "")
            return tf.strings.regex_replace(
                remove_special_chars, f"[{re.escape(string.punctuation)}]", ""
            )

        with tf.keras.utils.custom_object_scope({'custom_standardization': custom_standardization}):
            model = tf.keras.models.load_model(self.model_path)
            logging.info(f'Model Successfully Loaded!')

        return model
