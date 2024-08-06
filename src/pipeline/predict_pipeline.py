import sys 
import pandas as pd
import numpy as np
from exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
      try:  
        model_path='artifact\model.pkl'
        preprocessor_path='artifact\preprocessor.pkl'


        preprocessor=load_object(preprocessor_path)
        model=load_object(model_path)

        data_scaled=preprocessor.transform(features)

        prediction=model.predict(data_scaled)
        
        return prediction
      
      except Exception as e:
         raise CustomException(e,sys)
      

class CustomData:
   def __init__(self,):
      pass
