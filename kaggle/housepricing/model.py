

# %%
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")

# %%
train = pd.read_csv('./kaggle/housepricing/input/train_feat.csv')
test = pd.read_csv('./kaggle/housepricing/input/test_feat.csv')

# %%
number_sales = train.groupby('Neighborhood')[['Id']].count().rename(
    columns={'Id':'NumberSales'}).reset_index()
train = train.merge(number_sales,how='left',on='Neighborhood')
test = test.merge(number_sales,how='left',on='Neighborhood')

# %%
mean_price = train.groupby('Neighborhood')[['SalePrice']].mean().rename(
    columns={'SalePrice':'MeanPrice'}).reset_index()
train = train.merge(mean_price,how='left',on='Neighborhood')
test = test.merge(mean_price,how='left',on='Neighborhood')

# %%
train.head()
# %%
train.drop(columns=['Id'], inplace=True)

# %%
test_id = test['Id']
test = test.drop(columns=['Id'])

# %%
train['Years'] = train['YrSold'] - train['YearBuilt']
test['Years'] = test['YrSold'] - test['YearBuilt']

#%%
y_train_encode = train['SalePrice']
train.drop(columns=['SalePrice'],inplace=True)
# %%
dtypes = test.dtypes
numeric_categorical_cols = [
    'MSSubClass', 'OverallQual', 'OverallCond', 'MoSold'
    # 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'
]
numeric_cols = [col for col in dtypes[dtypes != "object"].index.values
                if col not in numeric_categorical_cols]
numeric_KBinsDiscretizer = KBinsDiscretizer(n_bins=10)

# %%
categorical_cols = test.dtypes[test.dtypes == "object"].index
categorical_oneHotEncoder = OneHotEncoder(handle_unknown='ignore')
numeric_categorical_oneHotEncoder = OneHotEncoder(handle_unknown='ignore')

# %%
categorical_dict = {}
freq_threshold = train.shape[0] * 0.01
for col in [*numeric_categorical_cols, *categorical_cols]:
    value_counts = train[col].value_counts()
    col_values = value_counts.index.values[value_counts > freq_threshold]
    categorical_dict[col] = {value: code for code,
                             value in enumerate(col_values, 1)}
    train[col] = train[col].apply(
        lambda x: categorical_dict[col][x] if x in col_values else 0)

    test[col] = test[col].apply(
        lambda x: categorical_dict[col][x] if x in col_values else 0)

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num_cat', numeric_categorical_oneHotEncoder,
            numeric_categorical_cols),
        ('num', numeric_KBinsDiscretizer, numeric_cols),
        ('cat', categorical_oneHotEncoder, categorical_cols)],
)
preprocessor.fit(train)

# %%
X_train_encode = preprocessor.transform(train)
X_test = preprocessor.transform(test)

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_train_encode, y_train_encode, test_size=0.2, random_state=42)
# %%
lr = Ridge(max_iter=2000)
lr.fit(X_train, y_train)
print('------lr------')
print(np.sqrt(mean_squared_error(y_val, lr.predict(X_val))))

# %%
from fastFM import als
fm = als.FMRegression(n_iter=1000, init_stdev=0.1, l2_reg_w=0.1, l2_reg_V=0.5)
fm.fit(X_train, y_train)
print('------fm------')
print(np.sqrt(mean_squared_error(y_val, fm.predict(X_val))))
# %% gbdt
X_train, X_val, y_train, y_val = train_test_split(
    train, y_train_encode, test_size=0.2, random_state=42)
# %%
feature_name = test.columns.tolist()
dataset_train = lgb.Dataset(pd.DataFrame(
    data=X_train, columns=feature_name), y_train)
dataset_test = lgb.Dataset(pd.DataFrame(
    data=X_val, columns=feature_name), y_val)
categorical_cols_index = [index for index, col in enumerate(feature_name)
                          if index in [*numeric_categorical_cols, *categorical_cols]]
params = {
    'objective': 'regression_l2',
    'boosting': 'gbdt',
    'metric': 'root_mean_squared_error',
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_seed': 1234,
    'feature_fraction': 0.8,
    'categorical_feature': categorical_cols_index,
    'random_state': 1234
}

lightgbm = lgb.train(params, dataset_train, num_boost_round=1000,
                     valid_sets=dataset_test, early_stopping_rounds=50)
print(mean_squared_error(y_val, lightgbm.predict(X_val)))
# %%
y_test_ = np.expm1(lr.predict(X_test))
submission = pd.DataFrame(
    {
        "Id": test_id,
        "SalePrice": y_test_
    }
)
# %%
submission.to_csv('./kaggle/housepricing/input/lightgbm.csv', index=False)

# %%
