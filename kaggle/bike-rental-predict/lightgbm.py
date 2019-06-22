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
def objective_func(train_set,num_boost_round=10000,nfold=5,stratified=False,shuffle=True,early_stopping_rounds=50,seed=1234):
    def objective(params):
        ret = lgb.cv(params,dataset_train,num_boost_round=10000,nfold=5,stratified=False,shuffle=True,early_stopping_rounds=50,seed=1234)
        return ret['rmse-mean'][-1]
    return objective

space={
    'objective':'regression_l2',
    'boosting':'gbdt',
    'metric':'root_mean_squared_error',
    'max_depth':hp.choice('max_depth', np.linspace(4,10,7).astype(np.int)),
    'num_leaves': hp.choice('num_leaves', np.linspace(32,128,49).astype(np.int)),
    'bagging_fraction':0.8,
    'bagging_freq':5, 
    'bagging_seed':1234,
    'categorical_feature':[0,1,2,3,4,5,6],
    'feature_fraction':0.8,
    'learning_rate': hp.uniform('learning_rate',0,0.3),
    'random_state':1234,
    'lambda_l2':hp.uniform('lambda_l2',0,20)
}
dataset_train = lgb.Dataset(X_train, y_train)
best = fmin(fn=objective_func(dataset_train,num_boost_round=10000,nfold=5,stratified=False,shuffle=True,early_stopping_rounds=50,seed=1234),
            space=space,
            algo=tpe.suggest,
            max_evals=300)
#%%
print(best)

#%%
space={
    'objective':'regression_l2',
    'boosting':'gbdt',
    'metric':'root_mean_squared_error',
    'bagging_fraction':0.8,
    'bagging_freq':5, 
    'bagging_seed':1234,
    'categorical_feature':[0,1,2,3,4,5,6],
    'feature_fraction':0.8,
    'learning_rate': hp.uniform('learning_rate',0,1),
    'random_state':1234,
    'lambda_l2':hp.uniform('lambda_l2',0,20)
}
space.update(best)
print(space)
#%%
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
dataset_train = lgb.Dataset(X_train_,y_train_)
dataset_test = lgb.Dataset(X_test_,y_test_)
lightgbm = lgb.train(space,dataset_train,num_boost_round=10000,valid_sets=dataset_test,early_stopping_rounds=50)
y_train_predicted = lightgbm.predict(X_train)
print("RMSLE on train: ", np.sqrt(mean_squared_error(y_train_predicted, y_train)))
#%%
print(list(zip(lightgbm.feature_importance(),lightgbm.feature_name())))

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
X_test = data_test[['season', 'holiday', 'workingday', 'weather','month','hour','dayofweek','temp','atemp','humidity','windspeed']]
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