import numpy as np
import pandas as pd

def prob_to_class(y_pre_prob,thresold=0.5):
    return np.where(y_pre_prob>=thresold,1,0)

def preprocess_user_input(raw_data, scaler):
    df=pd.DataFrame([raw_data])
    
    df['Fare']=np.log1p(df['Fare'])
    df['FamilySize']=df['SibSp']+df['Parch']+1
    df['Pclass_2']=df['Pclass'].apply(lambda x: 1 if x==2 else 0)
    df['Pclass_3']=df['Pclass'].apply(lambda x: 1 if x==3 else 0)
    df['Sex_male'] = df['Sex'].apply(lambda x: 1 if x == 'Nam' else 0)
    df['Embarked_Q']=df['Embarked'].apply(lambda x: 1 if x=='Queenstown (Q)' else 0)
    df['Embarked_S']=df['Embarked'].apply(lambda x: 1 if x=='Southampton (S)' else 0)
    df['FamilyType_Small']=df['FamilySize'].apply(lambda x: 1 if 2<=x and x<=4 else 0)
    
    expected_columns=['Age','Fare','FamilySize','Pclass_2','Pclass_3','Sex_male','Embarked_Q',
                      'Embarked_S','FamilyType_Small']
    df_final = df[expected_columns]
    cleaned_array=scaler.transform(df_final)
    return cleaned_array