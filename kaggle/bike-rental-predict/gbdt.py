# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error

import lightgbm as lgb

from hyperopt import hp, tpe, space_eval, STATUS_OK, Trials
from hyperopt.fmin import fmin

import warnings
warnings.filterwarnings("ignore")
# %%
use_cols = ['season', 'holiday', 'workingday', 'weather', 'temp',
            'atemp', 'humidity', 'windspeed', 'month', 'hour', 'dayofweek']
data_train = pd.read_csv(
    './kaggle/bike-rental-predict/input/train_feature.csv')
data_train.shape
# %%
data_train.head(3)
# %%
y_train = data_train['y'].values
X_train = data_train[use_cols]
# %%


def objective_fun(dataset_train, model_params={}, fit_params={}):
    def lgbm(params):
        model_params_copy = model_params.copy()
        model_params_copy.update(params)
        ret = lgb.cv(model_params_copy, dataset_train, **fit_params)
        best_rounds = np.argmin(ret['rmse-mean'])
        return {
            'loss': ret['rmse-mean'][best_rounds],
            'status': STATUS_OK,
            'loss_variance': ret['rmse-stdv'][best_rounds],
            'best_rounds': best_rounds
        }
    return lgbm


# %%
model_params = {
    'objective': 'regression_l2',
    'boosting': 'gbdt',
    'metric': 'root_mean_squared_error',
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_seed': 1234,
    'feature_fraction': 0.8,
    'categorical_feature': [0, 1, 2, 3, 4, 5, 6],
    'random_state': 1234
}


fit_params = {
    'num_boost_round': 1000,
    'nfold': 5,
    'stratified': False,
    'shuffle': True,
    'early_stopping_rounds': 50,
    'seed': 1234
}

space = {
    'max_depth': hp.choice('max_depth', np.linspace(4, 10, 7).astype(np.int)),
    'num_leaves': hp.choice('num_leaves', np.linspace(32, 128, 49).astype(np.int)),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1, 0.2]),
    'lambda_l2': hp.uniform('lambda_l2', 0, 20)
}
trials = Trials()
dataset_train = lgb.Dataset(X_train, y_train)
best = fmin(fn=objective_fun(dataset_train, model_params, fit_params),
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=300)
# %%
print(space_eval(space, best))
# %%
X_train_, X_test_, y_train_, y_test_ = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)
dataset_train = lgb.Dataset(X_train_, y_train_)
dataset_test = lgb.Dataset(X_test_, y_test_)
params = model_params.copy()
params.update(space_eval(space, best))
lightgbm = lgb.train(params, dataset_train, num_boost_round=1000,
                     valid_sets=dataset_test, early_stopping_rounds=50)
y_train_predicted = lightgbm.predict(X_train)
print("RMSLE on train: ", np.sqrt(mean_squared_error(y_train_predicted, y_train)))
# %%
print(list(zip(lightgbm.feature_importance(), lightgbm.feature_name())))

# %%
data_test = pd.read_csv("./kaggle/bike-rental-predict/input/test_feature.csv")
data_test_ids = data_test["datetime"]
data_test.head()

# %%
X_test = data_test[use_cols]
# %%
y_test_predicted = lightgbm.predict(X_test)
y_test_predicted = np.exp(y_test_predicted) - 1
# %%
submission = pd.DataFrame({
    "datetime": data_test_ids,
    "count": np.rint(y_test_predicted).astype(np.int)
})
submission.head()
# %%
submission.to_csv(
    './kaggle/bike-rental-predict/input/bike_predictions.csv', index=False)
