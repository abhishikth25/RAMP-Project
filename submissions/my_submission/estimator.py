from pathlib import Path

import numpy as np
import pandas as pd
from math import pi
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    
    cols = ['date','weighted_mob','Temp','Wind','Rain','Cloud','3h_car_count',
            '1h_car_count','new_cases','total_vaccinations','stringency_index',
            '500m construction','1000m construction']
    X = pd.merge_asof(X.sort_values('date'), df_ext[cols].sort_values('date'), on='date')
    # Sort back to the original order
    X = X.sort_values('orig_index')
    X[cols[8:10]] = X[cols[8:10]].fillna(0)
    del X['orig_index']
    return X

def get_estimator():
    
    date_encoder = FunctionTransformer(_encode_dates)
    #date_cols = _encode_dates(X_train[['date']]).columns.tolist()
    date_cols = ['year', 'month', 'day']

    categorical_encoder = OneHotEncoder(handle_unknown='use_encoded_value')
    categorical_cols = ["counter_name", "site_name","weekday","hour"]

    numeric_cols = ['weighted_mob','Temp','Wind','Rain','Cloud',
                    '3h_car_count','1h_car_count','new_cases','total_vaccinations',
                    'stringency_index','500m construction','1000m construction']

    preprocessor = ColumnTransformer([
        ('date', "passthrough", date_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ('numeric', 'passthrough', numeric_cols)
    ])

    #regressor = RandomForestRegressor(criterion="squared_error",n_estimators=250,max_depth=13)
    #regressor = XGBRegressor(objective='reg:squarederror',verbosity=0)
    regressor = HistGradientBoostingRegressor(loss='squared_error')
    #regressor = GradientBoostingRegressor(loss='squared_error')
    
    paramsRF = dict(n_estimators=[100,250],max_depth=[13,18])
    paramsXGB = dict(learning_rate=[0.05,0.1,0.15,0.2], max_depth=[2,4,6,8,10], 
                     n_estimators=[100,150,200,250], subsample=[0.4,0.55,0.7,1])
    paramsHGB = dict(max_iter=[200,225,250,275,350],max_depth=[4,6,8,10,12],
                     learning_rate=[0.1,0.2,0.25,0.3],l2_regularization=[0,0.05,0.1,0.15])
    paramsGB = dict(n_estimators=[150,200,250,350],max_depth=[4,6,8,10],
                    subsample=[0.4,0.55,0.7],learning_rate=[0.05,0.1,0.15,0.2])
    
    pipe =  make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        #regressor
        #GridSearchCV(regressor, paramsHGB, cv=2, n_jobs=-1, refit=True)
        RandomizedSearchCV(regressor, paramsHGB, n_iter=15, n_jobs=-1, random_state=1, refit=True)
        #BayesSearchCV(regressor, paramsHGB, n_iter=69, n_jobs=-1, refit=True, scoring='neg_mean_squared_error')
    )
    
    return pipe