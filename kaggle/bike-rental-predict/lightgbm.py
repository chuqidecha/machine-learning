#%%
import numpy as np 
import pandas as pd 
from sklearn.model_selection import GridSearchCV,KFold
from sklearn import metrics
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
#%%
data_train = pd.read_csv('./kaggle/bike-rental-predict/input/train.csv')
data_train.shape

#%%
data_train.head(3)
#%%
data_train.isnull().values.any()
#%%
data_train.datetime = data_train.datetime.apply(pd.to_datetime)
data_train['month'] = data_train.datetime.dt.month
data_train['hour'] = data_train.datetime.dt.hour
data_train['day'] = data_train.datetime.dt.day
data_train['dayofweek'] = data_train.datetime.dt.dayofweek
data_train['y'] = np.log(data_train['count'] + 1)
data_train.head()
#%%
y_train = data_train['y'].values
X_train = data_train[['season', 'holiday', 'workingday', 'weather','month','hour','dayofweek','temp','atemp','humidity','windspeed']]
print(list(zip(X_train.columns,X_train.dtypes)))
#%%
lightgbm = lgb.LGBMRegressor(objective='regression_l2',
                       boosting='gbdt',
                       metric='root_mean_squared_error',
                       early_stopping_round=20
                       bagging_fraction=0.8,
                       bagging_freq=5, 
                       bagging_seed=1234,
                       categorical_feature = 'name:season,holiday,workingday,weather,month,hour,dayofweek',
                       random_state=1234)
#%%
kf = KFold(n_splits=5, random_state=1234, shuffle=True)
#%% 学习率和迭代次数调参
param_grid={'n_estimators':[50,100,200,300,400,500],
            'learning_rate':np.logspace(-5,0,5)}
grid_search = GridSearchCV(lightgbm,param_grid=param_grid,scoring='neg_mean_squared_error',cv=kf) 
grid_search.fit(X_train,y_train)
print("Best parameters:{}".format(grid_search.best_params_))
print("Best best_score:{}".format(-grid_search.best_score_))
#%%
pd.DataFrame(grid_search.cv_results_).head(20)
#%%
y_train_predicted =  lightgbm.predict(X_train)
print("RMSLE on train: ", np.sqrt(metrics.mean_squared_error(y_train_predicted, y_train)))
#%%
data_test = pd.read_csv("./kaggle/bike-rental-predict/input/test.csv")
data_test_ids = data_test["datetime"]
data_test.head()

#%%
data_test.datetime = data_test.datetime.apply(pd.to_datetime)
data_test['month'] = data_test.datetime.dt.month
data_test['hour'] = data_test.datetime.dt.hour
data_test['dayofweek'] = data_test.datetime.dt.dayofweek
data_test.head()
#%%
X_test = data_test.drop(['datetime'], axis = 1)
#%%
y_test_predicted = grid_search.best_estimator_.predict(X_test)
y_test_predicted = np.exp(y_test_predicted) - 1
#%%
submission = pd.DataFrame({
        "datetime": data_test_ids,
        "count": np.rint(y_test_predicted).astype(np.int)
    })
submission.head()
#%%
submission.to_csv('./kaggle/bike-rental-predict/input/bike_predictions.csv', index=False)

#%%
