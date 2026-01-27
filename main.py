from src.cnnClassifier.logger import logging
from src.cnnClassifier.customexception import CustomException
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_model import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline
import sys


# STAGE_NAME = "Data Ingestion stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.main()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
    
#   raise CustomException(e,sys)


# STAGE_NAME = "Prepare base model"
# if __name__ == '__main__':
#     try:
#         logging.info(f"*******************")
#         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = PrepareBaseModelTrainingPipeline()
#         obj.main()
#         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logging.exception(e)
#         raise e


# STAGE_NAME = "Training"
# if __name__ == '__main__':
#     try:
#         logging.info(f"*******************")
#         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = ModelTrainingPipeline()
#         obj.main()
#         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logging.exception(e)
#         raise e



STAGE_NAME = "Evaluation stage"
try:
   logging.info(f"*******************")
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logging.exception(e)
        raise e