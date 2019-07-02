"""
数据预处理
"""


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew

import warnings
warnings.filterwarnings("ignore")

# %%
train = pd.read_csv('./kaggle/housepricing/input/train_fill.csv')
test = pd.read_csv('./kaggle/housepricing/input/test_fill.csv')

# %%

# 异常值处理
fig, ax = plt.subplots(figsize=(12,8),ncols=2,nrows=1)
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice',ax=ax[0])
sns.distplot(train['SalePrice'] , fit=norm, ax=ax[1])
plt.show()
# %%
