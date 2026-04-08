from src.load import load_data
from src.info import basic_info
from src.preprocess import handle_missing_data
from src.visualize import (
    plot_hist, box_plot, handle_outlier, 
    preprocess, get_important_features, corr_relation, scatter_plot)
from src.model import train_model, predict, evaluate, train_random_forest, evaluate_rf_forest
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import pandas as pd


# Step 1: Load
df = load_data("data/processed/train.csv")

# Step 2: Info
if df is not None:
    basic_info(df)

    # Step 3: Clean
    df = handle_missing_data(df)

    df = preprocess(df, 'saleprice')
    #Before outlier
    print("\n Before outlier handling")
    plot_hist(df, 'saleprice')
    box_plot(df, 'saleprice')

    #step 4: handle outlier
    df = handle_outlier(df, col='saleprice')

    #After outlier
    print("\n After outlier handling")
    plot_hist(df, 'saleprice')
    box_plot(df, 'saleprice')

    #  Fix target type
    df['saleprice'] = pd.to_numeric(df['saleprice'], errors='coerce')

    #Remove invalid target
    df = df[df['saleprice'] > 0]

    #Feature selection (Day 3)
    df = get_important_features(df)

    #Split
    X = df.drop('saleprice', axis=1)
    y = df['saleprice']
    
    corr_relation(df)
    #scatter_plot(df, 'area', 'saleprice')
    print("\n✅ Final Data:")
    print(df.head(10))
    print("Final Shape:", df.shape)
    print("Features used:", X.columns)

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("\n Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

#Train Linear
model = train_model(X_train, y_train)

#Predict linear
y_predict = model.predict(X_test)

#Evaluate linear
evaluate(y_test, y_predict)

#Train RandomForest
rf_model = train_random_forest(X_train, y_train)

#Predict RandomForest
rf_predict = rf_model.predict(X_test)

#Evaluate RandomForest
evaluate_rf_forest(y_test, rf_predict)

#feature importance
importance = rf_model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)
print('\n Feature Importance:\n', feature_importance_df)

#Cross validate
scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
print('\n cross validation scores', scores)
print('\n Average scores:', scores.mean())

#hyper tunning grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2,5]
}

rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf,
                    param_grid, 
                    cv=3, 
                    scoring='r2',
                    n_jobs=-1
)
grid.fit(X_train, y_train)
print("\n Best parameters:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Tuning model R2:", r2_score(y_test, y_pred)) 

joblib.dump(best_model, "house_price_model.pkl")