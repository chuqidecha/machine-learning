"""
数据预处理
"""

# %%
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# %%
# 加载训练数据
data_train = pd.read_csv('./kaggle/bike-rental-predict/input/train.csv')
data_train.shape
# %%
data_train.head(3)
# %%
data_train.isnull().values.any()
# %% 特征处理
data_train.datetime = data_train.datetime.apply(pd.to_datetime)
data_train['year'] = data_train.datetime.dt.year
data_train['month'] = data_train.datetime.dt.month
data_train['hour'] = data_train.datetime.dt.hour
data_train['day'] = data_train.datetime.dt.day
data_train['dayofweek'] = data_train.datetime.dt.dayofweek
data_train['y'] = np.log(data_train['count'] + 1)
data_train.head()

# %%
# 保存训练数据
data_train.to_csv(
    './kaggle/bike-rental-predict/input/train_feature.csv', index=False)

# %%
# 测试数据
data_test = pd.read_csv("./kaggle/bike-rental-predict/input/test.csv")
data_test.head()

# %%
data_test.datetime = data_test.datetime.apply(pd.to_datetime)
data_test['year'] = data_test.datetime.dt.year
data_test['month'] = data_test.datetime.dt.month
data_test['hour'] = data_test.datetime.dt.hour
data_test['dayofweek'] = data_test.datetime.dt.dayofweek
data_test.head()

# %%
data_test.to_csv(
    './kaggle/bike-rental-predict/input/test_feature.csv', index=False)
