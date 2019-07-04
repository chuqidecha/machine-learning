"""
缺失值填充
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
train['Shed'] = train['Shed'].fillna(value=0)
# %%
misc_data = test[test['MiscFeature'] == 'Shed'][['MiscFeature', 'MiscVal']]
misc_data = misc_data.pivot(columns='MiscFeature', values='MiscVal')
test = test.join(misc_data).drop(['MiscFeature', 'MiscVal'], axis=1)
test['Shed'] = test['Shed'].fillna(value=0)
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
    'GarageType',  'GarageFinish', 'GarageQual', 'GarageCond'
]
# garage_columns should be NA when no garage
pd.DataFrame([train[garage_columns].isna().sum(),
              test[garage_columns].isna().sum()])

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
    'GarageFinish': garage_type_group['GarageFinish'].agg(lambda x: x.mode()[0]),
    'GarageQual': garage_type_group['GarageQual'].agg(lambda x: x.mode()[0]),
    'GarageCond': garage_type_group['GarageCond'].agg(lambda x: x.mode()[0]),
    'GarageCars': garage_type_group['GarageCars'].agg(lambda x: x.mode()[0]),
    'GarageArea': garage_type_group['GarageArea'].agg(lambda x: x.median())
}

# %%
for column, fill_dict in garage_fill_dict.items():
    index = (test['GarageType'].notna() & test[column].isna())
    test.loc[index, column] = test.loc[index, 'GarageType'].apply(
        lambda x: fill_dict[x])

# %%
# GarageYrBlt如果没有用房子建造的时间填充
train_na_GarageYrBlt = train['GarageYrBlt'].isna()
train.loc[train_na_GarageYrBlt,
          'GarageYrBlt'] = train.loc[train_na_GarageYrBlt, 'YearBuilt']
test_na_GarageYrBlt = test['GarageYrBlt'].isna()
test.loc[test_na_GarageYrBlt,
         'GarageYrBlt'] = test.loc[test_na_GarageYrBlt, 'YearBuilt']
# %%
train[garage_columns] = train[garage_columns].fillna(value='None')
test[garage_columns] = test[garage_columns].fillna(value='None')

# %%
train[garage_columns].isna().sum()
test[garage_columns].isna().sum()

# %%
# 地下室相关特征
bsmt_columns = [
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'
]
train[bsmt_columns].head(10)
# %%
pd.DataFrame([train[bsmt_columns].isna().sum(),
              test[bsmt_columns].isna().sum()])
# %%
# BsmtQual、BsmtCond、BsmtExposure如果有一个不为NA则表示有地下室，其他的用众数填充
bsmt_cat_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure']
train_na = train[bsmt_cat_columns].isna().sum(axis=1)
index = train_na.gt(0) & train_na.lt(3)
train.loc[index, bsmt_columns].head(20)
# %%
test_na = test[bsmt_cat_columns].isna().sum(axis=1)
index = test_na.gt(0) & test_na.lt(3)
test.loc[index, bsmt_columns].head(20)

# %%
# 测试集中有7行，训练集中有一行，需要用众数填充
for column in bsmt_cat_columns:
    # 先填充测试集再填充训练集，防止训练集填充时众数被填充值取代了
    # 测试集上的填充应该用训练集上的众数，因为测试数据是不可见的
    test_index = test_na.ne(0) & test_na.ne(3) & test[column].isnull()
    test.loc[index, column] = train[column].mode()[0]
    train_index = train_na.ne(0) & train_na.ne(3) & train[column].isnull()
    train.loc[train_index, column] = train[column].mode()[0]

# %%
# 剩下的为空时是因为没有地下室，用“None”填充，面积相关的几个用0填充
bmst_na_columns = ['BsmtQual', 'BsmtCond',
                   'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
train[bmst_na_columns] = train[bmst_na_columns].fillna(value='None')
test[bmst_na_columns] = test[bmst_na_columns].fillna(value='None')
test[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
      'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0

# %%
# PoolQC、Alley、Fence、FireplaceQu 为NA时表示没有，用“None”填充
columns = ['PoolQC', 'Alley', 'Fence', 'FireplaceQu']
train[columns] = train[columns].fillna(value='None')
test[columns] = test[columns].fillna(value='None')

# %%
# MasVnrType、MasVnrArea
pd.DataFrame([train[['MasVnrType', 'MasVnrArea']].isna().sum(),
              test[['MasVnrType', 'MasVnrArea']].isna().sum()])

# %%
#  MasVnrType、MasVnrArea一致性检查（训练集）
index = train[['MasVnrType', 'MasVnrArea']].isnull().sum(axis=1)
train.loc[index.eq(1), ['MasVnrType', 'MasVnrArea']].head()

# %%
# MasVnrType、MasVnrArea一致性检查（测试集）
index = test[['MasVnrType', 'MasVnrArea']].isnull().sum(axis=1)
test.loc[index.eq(1), ['MasVnrType', 'MasVnrArea']].head()
# %%
# 测试集上不一致的使用众数填充
test.loc[index.eq(1), 'MasVnrType'] = train['MasVnrType'].mode()[0]

# %%
# MasVnrType用None填充，MasVnrArea用0填充
train['MasVnrType'] = train['MasVnrType'].fillna(value='None')
test['MasVnrType'] = test['MasVnrType'].fillna(value='None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(value=0)
test['MasVnrArea'] = test['MasVnrArea'].fillna(value=0)

# %%
# KitchenQual与KitchenAbvGr一致性检测
test.loc[test['KitchenQual'].isnull() & test['KitchenAbvGr'].gt(0), [
    'KitchenQual', 'KitchenAbvGr']].head()
# %%
# KitchenQual有一个NA，用众数填充
test.loc[test['KitchenQual'].isnull(
), 'KitchenQual'] = train['KitchenQual'].mode()[0]

# %%
# MSZoning、Utilities、Electrical邻近社区基本相同，用邻近社区众数填充
neighborhood_grouped = train.groupby('Neighborhood')
mode_fill_dict = {
    'MSZoning': neighborhood_grouped['MSZoning'].apply(lambda x: x.mode()[0]),
    'Utilities': neighborhood_grouped['Utilities'].apply(lambda x: x.mode()[0]),
    'Electrical':  neighborhood_grouped['Electrical'].apply(lambda x: x.mode()[0])
}

# %%
for column, fill_dict in mode_fill_dict.items():
    test.loc[test[column].isnull(), column] = test.loc[
        test[column].isnull(), 'Neighborhood'].apply(lambda x: fill_dict[x])
    train.loc[train[column].isnull(), column] = train.loc[
        train[column].isnull(), 'Neighborhood'].apply(lambda x: fill_dict[x])

# %% LotFrontage用邻近社区中位数填充
lot_frontage = train.groupby("Neighborhood")[
    "LotFrontage"].apply(lambda x: x.median())

train.loc[train['LotFrontage'].isnull(), 'LotFrontage'] = train.loc[
    train['LotFrontage'].isnull(), 'Neighborhood'].apply(lambda x: lot_frontage[x])
test.loc[test['LotFrontage'].isnull(), 'LotFrontage'] = test.loc[
    test['LotFrontage'].isnull(), 'Neighborhood'].apply(lambda x: lot_frontage[x])

# %%
# Functional、Exterior2nd、Exterior1stz用众数填充
test.loc[test['SaleType'].isnull(), 'SaleType'] = train['SaleType'].mode()[0]
test.loc[test['Functional'].isnull(), 'Functional'] = train['Functional'].mode()[0]
test.loc[test['Exterior2nd'].isnull(
), 'Exterior2nd'] = train['Exterior2nd'].mode()[0]
test.loc[test['Exterior1st'].isnull(
), 'Exterior1st'] = train['Exterior1st'].mode()[0]

# %%
na_train = train.isna().sum()
na_test = test.isna().sum()
print("-----------train----------")
print('%2d columns has na in train.' % (na_train.gt(0).sum()))
print(na_train[na_train.gt(0)].sort_values(ascending=False).to_dict())
print("-----------test----------")
print('%2d columns has na in test.' % (na_test.gt(0).sum()))
print(na_test[na_test.gt(0)].sort_values(ascending=False).to_dict())

# %%
# 保存结果
train.to_csv('./kaggle/housepricing/input/train_fill.csv', index=False)
test.to_csv('./kaggle/housepricing/input/test_fill.csv', index=False)

# %%
print("ALL DONE")
