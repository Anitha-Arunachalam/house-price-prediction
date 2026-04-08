from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Linear Regression start    
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    y_predict = model.predict(X_test)
    return y_predict

def evaluate(y_test, y_predict):
    score = r2_score(y_test, y_predict)
    print("R2 Score Linear:", score)
# linear regression end

#random forest
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_rf_forest(y_test, rf_predict):
    score = r2_score(y_test, rf_predict)
    print("R2 score Random forest:", score)

