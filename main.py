from src.cnnClassifier.logger import logging
from src.cnnClassifier.customexception import CustomException
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
import sys


STAGE_NAME = "Data Ingestion stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    
  raise CustomException(e,sys)