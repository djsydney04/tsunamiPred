import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def preprocess():
    df = pd.read_csv("hf://datasets/mnemoraorg/seismic-tsunami-event-linkage/raw_seismic_tsunami_events.csv")

    #reading features from dataframe (inputs)
    X = torch.tensor(df['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude'].values, dtype=torch.float32)
    #reading target from dataframe (outputs)
    y = torch.tensor(df['tsunami'].values, dtype = torch.float32).squeeze(1)

    #splitting data into training and testing sets
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)
    #returning training and testing sets
    return x_train,x_test,y_train,y_test
    