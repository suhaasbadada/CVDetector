import pandas as pd
import pickle

def handle_yes_no(json,key):
    json[key] = 1 if json[key] == 'Yes' else 0

def preprocess_readings_input_lifestyle(json):
    order = ['General_Health',
             'Checkup',
             'Exercise',
             'Heart_Disease',
             'Skin_Cancer',
             'Other_Cancer',
             'Depression',
             'Diabetes',
             'Arthritis',
             'Sex',
             'BMI',
             'Smoking_History',
             'Alcohol_Consumption',
             'Fruit_Consumption',
             'Green_Vegetables_Consumption',
             'FriedPotato_Consumption',
             'Age_min',
             'Age_max']
    
    general_health_dict={'Poor':0,'Fair':1,'Good':2,'Very Good':3,'Excellent':4}
    checkup_dict={'Never':0,'Within the past year':1,'Within the past 2 years':2,'Within the past 5 years':3,'5 or more years ago':4}
    
    #Calculating BMI
    json['BMI']=int(json['Weight']/(json['Height']*json['Height']))
    
    #Getting age ranges
    age_range=json['Age_range']
    json['Age_min'],json['Age_max']=age_range.split('-')
    
    #Handling Boolean keys
    yes_no_keys=['Exercise','Skin_Cancer','Other_Cancer','Depression','Diabetes','Arthritis','Smoking_History']
    
    for key in yes_no_keys:
        handle_yes_no(json,key)
    
    #Handling User Sex
    json['Sex'] = 1 if json['Sex'] == 'Male' else 0
    
    #Handling general health & checkup keys
    json['General_Health']=general_health_dict[json['General_Health']]
    json['Checkup']=checkup_dict[json['Checkup']]
    
    #Deleting dispensable keys
    del json['Height']
    del json['Weight']
    del json['Age_range']
    
    ordered_input = {key: json[key] for key in order if key in json}
    return ordered_input

with open("model\\lifestyle_scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

with open("model\\lbgm_model.pkl", 'rb') as file:
    lbgm = pickle.load(file)

def run_model(json,model=lbgm):
    df=pd.DataFrame.from_dict([json])
    obj_scaled=scaler.transform(df)
    predicted=model.predict(obj_scaled)[0]
    obj={}
    obj[str(round(predicted))]=predicted
    obj[str(abs(1-round(predicted)))]=1-predicted
    obj['class']=round(predicted)
    return obj
