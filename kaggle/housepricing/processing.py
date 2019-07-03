"""
数据预处理
"""


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")

# %%
train = pd.read_csv('./kaggle/housepricing/input/train_fill.csv')
test = pd.read_csv('./kaggle/housepricing/input/test_fill.csv')

# %%
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice')
plt.show()

# %%
train = train.drop(train[(train['GrLivArea'] > 4000) & (
    train['SalePrice'] < 300000)].index)

# %%
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice')
plt.show()
# %%
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

fig, ax = plt.subplots(figsize=(12, 8), ncols=2, nrows=1)
sns.distplot(train['SalePrice'], fit=norm, ax=ax[0])
ax[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(
    mu, sigma)], loc='best')
res = stats.probplot(train['SalePrice'], plot=ax[1])
plt.show()
# %%
train["SalePrice"] = np.log1p(train["SalePrice"])

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

fig, ax = plt.subplots(figsize=(12, 8), ncols=2, nrows=1)
sns.distplot(train['SalePrice'], fit=norm, ax=ax[0])
ax[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(
    mu, sigma)], loc='best')
res = stats.probplot(train['SalePrice'], plot=ax[1])
plt.show()

# %%
# 虽然是数值但实际为类别的特征转为str
cat_numeric_cols = [
    'MSSubClass', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'MoSold',
    'YrSold', 'GarageYrBlt'
]
train[cat_numeric_cols] = train[cat_numeric_cols].astype(str)
test[cat_numeric_cols] = test[cat_numeric_cols].astype(str)

# %%
# 删除id列
train.drop("Id", axis=1, inplace=True)
test_id = test["Id"]
test.drop("Id", axis=1, inplace=True)

# %%
numeric_feats = test.dtypes[train.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = train[numeric_feats].apply(
    lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness.head(10)

# %%
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(
    skewness.shape[0]))


skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    train[feat] = boxcox1p(train[feat], lam)
    test[feat] = boxcox1p(test[feat], lam)

# %%
train.to_csv('./kaggle/housepricing/input/train_feat.csv', index=False)
test.to_csv('./kaggle/housepricing/input/test_feat.csv', index=False)
# %%
numeric_features = test.dtypes[train.dtypes != "object"].index
numeric_standardScaler = StandardScaler()
numeric_KBinsDiscretizer = KBinsDiscretizer(n_bins=10)


categorical_features = test.dtypes[train.dtypes == "object"].index
categorical_oneHotEncoder = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_KBinsDiscretizer, numeric_features),
        ('cat', categorical_oneHotEncoder, categorical_features)],
)
preprocessor.fit(train)

# %%
X_train_encode = preprocessor.transform(train)
y_train_encode = train['SalePrice']
X_test = preprocessor.transform(test)
# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_train_encode, y_train_encode, test_size=0.2, random_state=42)
# %%
lr = Ridge(max_iter=2000)
lr.fit(X_train, y_train)
print(mean_squared_error(y_val, lr.predict(X_val)))
# %%
y_test_ = np.expm1(lr.predict(X_test))
submission = pd.DataFrame(
    {
        "Id": test_id,
        "SalePrice": y_test_
    }
)
# %%
submission.to_csv('./kaggle/housepricing/input/ridge2.csv', index=False)

# %%
train.dtypes

#%%
