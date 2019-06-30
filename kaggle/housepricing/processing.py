"""
数据预处理
"""


# %%
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# %%
train = pd.read_csv('./kaggle/housepricing/input/train.csv')
test = pd.read_csv('./kaggle/housepricing/input/test.csv')

print('train size: %d, %d. test size: %d, %d' % (*train.shape, *test.shape))

# %%
train.head(10)

# %%
test.head(3)

# %%
train['MiscFeature'].value_counts()
# %%
misc_data = train[['MiscFeature', 'MiscVal']].dropna()
misc_data = misc_data.pivot(columns='MiscFeature', values='MiscVal')
misc_data.head()

# %%
# 混合特征(MiscFeature)展开成Elev、Gar2、Othr、Othr、TenC5列
misc_data = train[train['MiscFeature'] == 'Shed'][['MiscFeature', 'MiscVal']]
misc_data = misc_data.pivot(columns='MiscFeature', values='MiscVal')
train = train.join(misc_data).drop(['MiscFeature', 'MiscVal'], axis=1)

# %%
misc_data = test[test['MiscFeature'] == 'Shed'][['MiscFeature', 'MiscVal']]
misc_data = misc_data.pivot(columns='MiscFeature', values='MiscVal')
test = test.join(misc_data).drop(['MiscFeature', 'MiscVal'], axis=1)

# %%
# 查看缺失值情况
na_train = train.isna().sum()
na_test = test.isna().sum()
print("-----------train----------")
print('%2d columns has na in train.' % (na_train.gt(0).sum()))
print(na_train[na_train.gt(0)].sort_values(ascending=False).to_dict())
print("-----------test----------")
print('%2d columns has na in test.' % (na_test.gt(0).sum()))
print(na_test[na_test.gt(0)].sort_values(ascending=False).to_dict())
# test中的缺失值比train更严重。
# 在真实环境中我们无法知道test集中哪些列有缺失值，因此需要假设每一列都可能出现缺失值。

# %%
# 车库相关特征处理
garage_columns = [
    'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'
]
# garage_columns should be NA when no garage
train_row_na_count = train[garage_columns].isna().sum(axis=1)
print(train_row_na_count[train_row_na_count.ne(0) & train_row_na_count.ne(5)])
test_row_na_count = test[garage_columns].isna().sum(axis=1)
print(test_row_na_count[test_row_na_count.ne(0) & test_row_na_count.ne(5)])

# %%
test.iloc[[666, 1116]][garage_columns].head(5)

# %% 检查数据的一致性，即没有车库时GarageCars和GarageArea是否为0
print(train[train['GarageType'].isna() & train['GarageCars'].ne(
    0) & train['GarageArea'].ne(0)][['GarageCars', 'GarageArea']].isna().sum())

print(train[train['GarageType'].notna() & train['GarageCars'].ne(
    0) & train['GarageArea'].ne(0)][['GarageCars', 'GarageArea']].isna().sum())

print(test[test['GarageType'].isna() & train['GarageCars'].ne(
    0) & test['GarageArea'].ne(0)][['GarageCars', 'GarageArea']].isna().sum())

print(test[test['GarageType'].notna() & train['GarageCars'].ne(
    0) & test['GarageArea'].ne(0)][['GarageCars', 'GarageArea']].isna().sum())

# %%
test.iloc[[666, 1116]][[*garage_columns, 'GarageCars', 'GarageArea']].head(5)
# %%
# garage相关特征值填充
# 测试集中，id=666，1116的样本GarageYrBlt、GarageFinish、
# GarageQual、GarageCond需要用GarageType相同的众数填充
garage_type_group = train.groupby('GarageType')
garage_fill_dict = {
    'GarageYrBlt': garage_type_group['GarageYrBlt'].agg(lambda x: x.mode()[0]),
    'GarageFinish': garage_type_group['GarageFinish'].agg(lambda x: x.mode()[0]),
    'GarageQual': garage_type_group['GarageQual'].agg(lambda x: x.mode()[0]),
    'GarageCond': garage_type_group['GarageCond'].agg(lambda x: x.mode()[0]),
    'GarageCars': garage_type_group['GarageCars'].agg(lambda x: x.mode()[0]),
    'GarageArea': garage_type_group['GarageArea'].agg(lambda x: x.median())
}

# %%


def garage_fill():
    for column, fill_dict in garage_fill_dict.items():
        index = (test['GarageType'].notna() & test[column].isna())
        test.loc[index, column] = test.loc[index, 'GarageType'].apply(
            lambda x: fill_dict[x])


garage_fill()

# %%
train[garage_columns] = train[garage_columns].fillna(value='None')
test[garage_columns] = test[garage_columns].fillna(value='None')

# %%
train[garage_columns].isna().sum()
test[garage_columns].isna().sum()
# %%
# 类别特征，根据数据说明，NA为没有，用“None”填充
none_fill_columns = [
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PoolQC', 'Fence'
]

for column in none_fill_columns:
    train[column] = "None"
    test[column] = "None"



# %%
# 类别特征，特征值都是有限的几个，缺失时用众数填充
mode_fill_columns = [
    'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1'，'Condition2', 'BldgType', 'HouseStyle',
    'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
    'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType',
    'SaleCondition'
]


# %%
bsmt = [
    'BsmtQual', 'BsmtCond', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
    'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'
]
train[bsmt].head(100)
# %%
train.head(3)

# %%
