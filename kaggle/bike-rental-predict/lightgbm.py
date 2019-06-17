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
data_train['dayofweek'] = data_train.datetime.dt.dayofweek
data_train['y'] = np.log(data_train['count'] + 1)
data_train.head()
#%%
y_train = data_train['y']
X_train = data_train[['season', 'holiday', 'workingday', 'weather','month','hour','dayofweek','temp','atemp','humidity','windspeed']]
print(list(zip(X_train.columns,X_train.dtypes)))

#%%
X_train.head(3)
#%%
lightgbm = lgb.LGBMRegressor(objective='regression_l2',
                       boosting='gbdt',
                       metric='root_mean_squared_error',
                       max_depth=8,
                       bagging_fraction=0.8,
                       bagging_freq=5, 
                       bagging_seed=1234,
                       categorical_feature='0,1,2,3,4,5,6',
                       random_state=1234)
#%%
kf = KFold(n_splits=5, random_state=1234, shuffle=True)

#%%
dataset_train = lgb.Dataset(X_train,y_train)
params={
    'objective':'regression_l2',
    'boosting':'gbdt',
    'metric':'root_mean_squared_error',
    'max_depth':5,
    'bagging_fraction':0.8,
    'bagging_freq':5, 
    'bagging_seed':1234,
    'categorical_feature':[0,1,2,3,4,5,6],
    'random_state':1234
}
ret = lgb.cv(params,dataset_train,num_boost_round=10000,nfold=5,stratified=False,shuffle=True,early_stopping_rounds=50,seed=1234)

#%%
print('num_boost_round = %s, best_iter = %s' %(len(ret['rmse-mean']),np.argmin(ret['rmse-mean'])))
#%%
lightgbm = lgb.train(params,dataset_train,num_boost_round=len(ret['rmse-mean']))

#%%
y_train_predicted = lightgbm.predict(X_train,np.argmin(ret))
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
y_test_predicted = lightgbm.predict(X_test)
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
