import os
import sys
from exception import CustomException
from logger import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from utils import save_object
from dataclasses import dataclass
from data_ingestion import DataIngestion,DataIngestionConfig

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,data):

        try:
            data=pd.read_csv(data)
            data.drop(columns="Name",inplace=True)

            data.drop(columns="Revenue",inplace=True)


            num_columns=[num_column for num_column in list(data.columns) if data[num_column].dtype!=object]

            cat_columns=[cat_column for cat_column in list(data.columns) if data[cat_column].dtype==object]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder())
                ]
            )
            
            logging.info(f"numerical columns: {num_columns}")

            logging.info(f"categorical columns: {cat_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_columns),
                    ("cat_pipeline",cat_pipeline,cat_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

            
    def initiate_data_transformation(self,train_path,test_path):
        try:    
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading Train and Test Data Completed")

            logging.info("Loaind Preprocessor Object")

            preprocessor_obj=self.get_data_transformer_object(data="C:\\RestaurantSalesPredictionEndtoEnd\\artifact\\data.csv")

            target_column="Revenue"

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("Input Features and Target Features Seperated for both Train and Test data")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                 input_feature_train_arr,target_feature_train_df
            ]

            test_arr=np.c_[
                 input_feature_test_arr,target_feature_test_df
            ]

            # save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path ,obj=preprocessor_obj)
            print(self.data_transformation_config.preprocessor_obj_file_path )
            return (
                 train_arr,
                 test_arr,
                 self.data_transformation_config.preprocessor_obj_file_path 

            )
        except Exception as e:
            raise CustomException(e,sys)



if __name__ =="__main__":
    obj=DataIngestion()    
    train_path, test_path = obj.initiate_data_ingestion()

    obj1=DataTransformation()
    obj1.get_data_transformer_object(data="C:\\RestaurantSalesPredictionEndtoEnd\\artifact\\data.csv")

    train_arr,test_arr,_=obj1.initiate_data_transformation(train_path=train_path,test_path=test_path)
    # print(train_arr)


    

