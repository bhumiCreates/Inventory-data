from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

def train_LR(X_train,Y_train):
    model = LinearRegression()
    model.fit(X_train,Y_train)
    return model

def train_DT(X_train,Y_train, max_depth=5):
    model=DecisionTreeRegressor(max_depth=max_depth , random_state=42)
    model.fit(X_train,Y_train)
    return model

def train_RF(X_train,Y_train,max_depth=6):
    model=RandomForestRegressor(max_depth=max_depth,random_state=42)
    model.fit(X_train,Y_train)
    return model

def evaluate_model(model, X_test,Y_test, model_name:str)-> dict:
    preds = model.predict(X_test)

    mae = mean_absolute_error(Y_test, preds)
    rmse = np.sqrt(mean_squared_error(Y_test, preds))
    r2 = r2_score(Y_test, preds) * 100

    return{
        "model_name":model_name,
        "mae":mae,
        "rmse":rmse,
        "r2":r2
    }