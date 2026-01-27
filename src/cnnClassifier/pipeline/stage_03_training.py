from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.model_training import ModelTrainer
from src.cnnClassifier.logger import logging



STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        
        training_config = config.get_training_config()
        training = ModelTrainer(config=training_config)
        training.train()
       




        