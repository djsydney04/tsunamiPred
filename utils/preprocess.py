import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def preprocess():
    ds = load_dataset("mnemoraorg/seismic-tsunami-event-linkage")   
    df = ds['train'].to_pandas()
    print("Available columns:", df.columns.tolist())

    #reading features from dataframe (inputs)
    X = torch.tensor(df[['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']].values, dtype=torch.float32)
    #reading target from dataframe (outputs)
    y = torch.tensor(df['tsunami'].values, dtype = torch.float32).unsqueeze(1)  
    #splitting data into training and testing sets
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)
    #returning training and testing sets
    return x_train,x_test,y_train,y_test
    

