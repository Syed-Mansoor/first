import os
import sys
from diamond.logger import logging
from diamond.exception.exception import DiamondException
import pandas as pd

from diamond.component.data_ingestion import DataIngestion
from diamond.component.data_transformation import DataTransformation
from diamond.component.model_trainer import ModelTrainer
from diamond.component.model_evaluation import ModelEvaluation


obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)


model_trainer_obj=ModelTrainer()
model_trainer_obj.initate_model_training(train_arr,test_arr)

model_evaluation=ModelEvaluation()
model_evaluation.initiate_model_evaluation(train_arr,test_arr)