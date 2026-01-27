from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml,create_directories,save_json
from src.cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                    PrepareBaseModelConfig,
                                                    TrainingConfig,EvaluationConfig)
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

   ## reading model
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
          ## preparing model


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            image_size=self.params.IMAGE_SIZE,
            learning_rate=self.params.LEARNING_RATE,
            classes=self.params.CLASSES,
            batch_size=self.params.BATCH_SIZE,
            epochs=self.params.EPOCHS,
            channel=self.params.CHANNELS,
            pretrained=self.params.PRETRAINED
        )

        return prepare_base_model_config
    

    ##  model training
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params

        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "mnist-m/training")

        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            training_data=Path(training_data),
            base_model_path=Path(
            self.config.prepare_base_model.base_model_path
        ),  
            epochs=params.EPOCHS,
            batch_size=params.BATCH_SIZE,
            learning_rate=params.LEARNING_RATE,
            image_size=params.IMAGE_SIZE,
        
        )

        return training_config
    
    ## model evaluation
        
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path("artifacts/training/trained_resnet18.pth"),
            training_data=Path("artifacts/data_ingestion/mnist-m/testing"),
            params_batch_size=self.params.BATCH_SIZE,
            num_classes = 10
        )
        return eval_config       