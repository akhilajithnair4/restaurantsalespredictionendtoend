import os
import sys
import pickle 
from exception import CustomException
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.join(file_path)

        os.makedirs(dir_path,exist_ok=True)
        # assert os.path.isfile(path)
        print("NO ERROR FOUND")
        with open(dir_path,'wb') as file:
            print("ERROR ERROR ERROR")
            pickle.dump(obj,file)

        

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,model_dict,params):
    try:
        report_dict={}
        for models in list(model_dict.keys()):
            reg_model=model_dict[models]
            
            para=params[models]
            # train the model

            gs = GridSearchCV(reg_model,para,cv=3)
            gs.fit(X_train,y_train)

            reg_model.set_params(**gs.best_params_)
            reg_model.fit(X_train,y_train)

            # reg_model.fit(X_train,y_train) 

            # predict on the training data
            reg_model.predict(X_train)
            # predict on the test data
            y_predicted=reg_model.predict(X_test)

            # calculate the r-sqaure
            r_sqaure=r2_score(y_test,y_predicted)

            report_dict[models]=r_sqaure
        
        report_df=pd.DataFrame(list(report_dict.items()),columns=['Model','R_Square'])
        final_report=report_df.sort_values(by='R_Square', ascending=False)
        return final_report

    except Exception as e:
       raise CustomException(e,sys)



    











