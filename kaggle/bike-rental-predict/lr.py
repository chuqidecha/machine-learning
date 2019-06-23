"""
线性回归
"""

#%%
import numpy as np 
import pandas as pd 
from sklearn.model_selection import GridSearchCV,KFold
from sklearn import linear_model,metrics,pipeline, preprocessing
import warnings
warnings.filterwarnings("ignore")
#%%
use_cols=['season', 'holiday', 'workingday', 'weather','temp','atemp','humidity','windspeed','month','hour','dayofweek']
data_train = pd.read_csv('./kaggle/bike-rental-predict/input/train_feature.csv')
data_train.shape
#%%
data_train.head(3)
#%%
y_train = data_train['y'].values
X_train = data_train[use_cols]
#%%
binary_data_columns = ['holiday', 'workingday']
binary_data_indices = np.array([(column in binary_data_columns) for column in X_train.columns], dtype = bool)

categorical_data_columns = ['season', 'weather', 'month', 'hour', 'dayofweek'] 
categorical_data_indices = np.array([(column in categorical_data_columns) for column in X_train.columns], dtype = bool)

numeric_data_columns = ['temp', 'atemp', 'humidity', 'windspeed']
numeric_data_indices = np.array([(column in numeric_data_columns) for column in X_train.columns], dtype = bool)
#%%
transformer_list = [        
            #binary
            ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])), 
                    
            #numeric
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                    ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),
                    ('scaling', preprocessing.StandardScaler(with_mean = 0)) 
                ])
            ),
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                    ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                    ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))
                ])
            ),
        ]
feature_encode = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list=transformer_list))]
)

X_train_encode = feature_encode.fit_transform(X_train)
#%%
X_train_encode.shape
#%%
data_test = pd.read_csv("./kaggle/bike-rental-predict/input/test_feature.csv")
data_test.head()
#%%
X_test = data_test[use_cols]
id_test = data_test['datetime']

#%%
X_test_encode = feature_encode.transform(X_test)
X_test_encode.shape
#%%
kf = KFold(n_splits=5, random_state=1234, shuffle=True)
#%%
estimator = linear_model.Ridge(max_iter=3000)
grid_search = GridSearchCV(estimator,
    param_grid={'alpha':np.logspace(-8,8,17)},
    scoring='neg_mean_squared_error',cv=kf) 
grid_search.fit(X_train_encode,y_train)
print("Best parameters:{}".format(grid_search.best_params_))
print("Best best_score:{}".format(-grid_search.best_score_))
#%%
pd.DataFrame(grid_search.cv_results_).head(20)
#%%
y_train_predicted =  grid_search.best_estimator_.predict(X_train_encode)
print("RMSLE on train: ", np.sqrt(metrics.mean_squared_error(y_train_predicted, y_train)))
#%%
y_test_predicted = grid_search.best_estimator_.predict(X_test_encode)
y_test_predicted = np.exp(y_test_predicted) - 1
#%%
submission = pd.DataFrame({
        "datetime": id_test,
        "count": np.rint(y_test_predicted).astype(np.int)
    })
submission.head()
#%%
submission.to_csv('./kaggle/bike-rental-predict/input/bike_predictions.csv', index=False)