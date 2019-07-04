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
# %%
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
numeric_feats = test.dtypes[train.dtypes == "float64"].index

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

#%%
