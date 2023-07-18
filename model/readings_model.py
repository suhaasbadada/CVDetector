import xgboost as xgb
import pandas as pd
import pickle
from fastapi.responses import JSONResponse

def preprocess_readings_input_readings(json):
    order=[ 'age',
            'ap_hi',
            'ap_lo',
            'cholesterol',
            'gluc',
            'physical_activity',
            'bmi',
            'gender']

    json['bmi']=json['weight']/(json['height']*json['height'])
    del json['height']
    del json['weight']  

    if json['bmi'] <= 15:
        json['bmi'] = 0
    elif json['bmi'] <= 18.5:
        json['bmi'] = 1
    elif json['bmi'] <= 25:
        json['bmi'] = 2
    elif json['bmi'] <= 30:
        json['bmi'] = 3
    elif json['bmi'] <= 35:
        json['bmi'] = 4
    elif json['bmi'] <= 40:
        json['bmi'] = 5
    else:
        json['bmi'] = 6
    
    json['gender'] = 1 if json['gender'] == 'Male' else 0
    json['physical_activity'] = 1 if json['physical_activity'] == 'Yes' else 0
    
    if json['cholesterol'] < 200:
        json['cholesterol'] = 1
    elif json['cholesterol'] < 240:
        json['cholesterol'] = 2
    else:
        json['cholesterol'] = 3
        
    if json['gluc'] < 5.6:
        json['gluc'] = 1
    elif json['gluc'] < 6.5:
        json['gluc'] = 2
    else:
        json['gluc'] = 3
    
    ordered_input = {key: json[key] for key in order if key in json}
    return ordered_input

with open("model\\readings_scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

with open("model\\xg_model.pkl", 'rb') as file:
    xg_model = pickle.load(file)

def xgboost(json,xg_model=xg_model):
    df=pd.DataFrame.from_dict([json])
    obj_scaled=scaler.transform(df)
    preds=xg_model.predict(xgb.DMatrix(obj_scaled))
    obj={'1':float(preds[0]),'0':float(1-preds[0]),'class':round(preds[0])}
    return obj
