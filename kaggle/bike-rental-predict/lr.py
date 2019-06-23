"""
线性回归
"""

#%%
import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_validate,GridSearchCV,KFold
from sklearn import linear_model,metrics,pipeline, preprocessing
import warnings
warnings.filterwarnings("ignore")
#%%
user_cols=['season', 'holiday', 'workingday', 'weather','temp','atemp','humidity','windspeed','month','hour','dayofweek']
data_train = pd.read_csv('./kaggle/bike-rental-predict/input/train_feature.csv')[[*user_cols,'y']]
data_train.shape
#%%
data_train.head(3)
#%%
y_train = data_train['y'].values
X_train = data_train[user_cols]
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
                        ])),
        
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ]

estimator = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list=transformer_list)),
    ('model_fitting', linear_model.Ridge(max_iter=2000))
    ]
)

#%%
kf = KFold(n_splits=5, random_state=1234, shuffle=True)

grid_search = GridSearchCV(estimator,
    param_grid={'model_fitting__alpha':np.logspace(-8,8,17),'model_fitting__max_iter',[500,1000,2000,3000]},
    scoring='neg_mean_squared_error',cv=kf) 
grid_search.fit(X_train,y_train)
print("Best parameters:{}".format(grid_search.best_params_))
print("Best best_score:{}".format(-grid_search.best_score_))
#%%
pd.DataFrame(grid_search.cv_results_).head(20)
#%%
y_train_predicted =  grid_search.best_estimator_.predict(X_train)
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