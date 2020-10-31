# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import os

PATH = "D:\projects\home_credit_regression\data"
# print(os.listdir(PATH))

# upload data
app_train = pd.read_csv("./data/application_train.csv")
app_test = pd.read_csv("./data/application_test.csv")

# %%
pd.set_option('display.max_columns', None)
# print(app_train.TARGET.value_counts())
# print(app_train.head())

# %%
# plot

plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = [8, 5]

plt.hist(app_train.TARGET)

plt.show()


# %%

def missing_value_table(df):
    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )

    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    print("В выбранном датафрейме " + str(df.shape[1]) + "столбцов.\n"
                                                         "Всего " + str(mis_val_table_ren_columns.shape[0]) +
          " столбцов с неполными данными."
          )

    return mis_val_table_ren_columns


missing_values = missing_value_table(app_train)
missing_values.head(10)

# %%

plt.style.use('seaborn-talk')

fig = plt.figure(figsize=(18, 6))
miss_train = pd.DataFrame((app_train.isnull().sum()) * 100 / app_train.shape[0]).reset_index()
miss_test = pd.DataFrame((app_test.isnull().sum()) * 100 / app_test.shape[0]).reset_index()
miss_train["type"] = "тренировочная"
miss_test["type"] = "тестовая"
missing = pd.concat([miss_train, miss_test], axis=0)
ax = sns.pointplot("index", 0, data=missing, hue="type")
plt.xticks(rotation=90, fontsize=7)
plt.title("Доля отсутствующих значений в данных")
plt.ylabel("Доля в %")
plt.xlabel("Столбцы")
plt.show()

# %%
app_train.dtypes.value_counts()
app_train.select_dtypes(include=[object]).apply(pd.Series.nunique, axis=0)

# %%
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# %%

train_labels = app_train['TARGET']

app_train, app_test = app_train.align(app_test, join='inner', axis=1)

print('Формат тренировочной выборки: ', app_train.shape)
print('Формат тестовой выборки: ', app_test.shape)

app_train['TARGET'] = train_labels

# %%

# correlation

correlations = app_train.corr()['TARGET'].sort_values()

print('Наивысшая позитивная корреляция: \n', correlations.tail(15))
print('/nНаивысшая негативная корреляция: \n', correlations.head(15))

# %% age

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])

plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
plt.title('Age of client');
plt.xlabel('Age (years)');
plt.ylabel('Count');
plt.show()

# %%

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label='target == 0')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label='target == 1')
plt.xlabel('Age (years');
plt.ylabel('Density');
plt.title('Distribution of Ages');
plt.show()

# %%

ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs

sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
plt.title('Correlation Heatmap')
plt.show()

# %%

plt.figure(figsize=(10, 12))

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    plt.subplot(3, 1, i + 1)

    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label='target == 0')

    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label='target == 1')

    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source)
    plt.ylabel('Density')

plt.tight_layout(h_pad=2.5)
plt.show()


# %%

application_train = pd.read_csv("./data/application_train.csv")
application_test = pd.read_csv("./data/application_test.csv")

def plot_stats(feature,label_rotation = False,horizontal_layout = True):
    temp = application_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Количество займов':temp.values})

    # Расчет доли target = 1 в категории

    cat_perc = application_train[[feature,'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by = 'TARGET', ascending  = False, inplace = True)

    if (horizontal_layout):
        fig,(ax1,ax2) = plt.subplots(ncols = 2,figsize = (12,6))
    else:
        fig,(ax1,ax2) = plt.subplots(ncols=2,figsize = (12,14))

    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Количество займов",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation = 90)

    s =sns.barplot(ax=ax2, x = feature , y = 'TARGET',order=cat_perc[feature],data=cat_perc)
    if label_rotation:
        s.set_xticklabels(s.get_xticklabels(),rotation = 90)

    plt.ylabel('Доля проблемных', fontsize = 10)
    plt.tick_params(axis='both',which = 'major',labelsize = 10)


    plt.show()

#%%
plot_stats('NAME_CONTRACT_TYPE')
plot_stats('CODE_GENDER')
plot_stats('FLAG_OWN_CAR')
plot_stats('FLAG_OWN_REALTY')
plot_stats('NAME_FAMILY_STATUS',True,True)
plot_stats('CNT_CHILDREN')
plot_stats('CNT_FAM_MEMBERS',True)
plot_stats('NAME_INCOME_TYPE',False,False)
plot_stats('OCCUPATION_TYPE',True, False)
plot_stats('NAME_EDUCATION_TYPE',True)
plot_stats('ORGANIZATION_TYPE',True, False)

#%%
plt.figure(figsize=(12,5))
plt.title("Распределение AMT_CREDIT")
ax = sns.distplot(app_train["AMT_CREDIT"])
plt.show()

#%%
plt.figure(figsize=(12,5))
# KDE займов, выплаченных вовремя
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'AMT_CREDIT'], label = 'target == 0')
# KDE проблемных займов
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'AMT_CREDIT'], label = 'target == 1')
# Обозначения
plt.xlabel('Сумма кредитования'); plt.ylabel('Плотность'); plt.title('Суммы кредитования');

plt.show()

#%%
plt.figure(figsize=(12,5))
plt.title("Распределение REGION_POPULATION_RELATIVE")
ax = sns.distplot(app_train["REGION_POPULATION_RELATIVE"])
plt.show()

#%%
plt.figure(figsize=(12,5))
# KDE займов, выплаченных вовремя
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'REGION_POPULATION_RELATIVE'], label = 'target == 0')
# KDE проблемных займов
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'REGION_POPULATION_RELATIVE'], label = 'target == 1')
# Обозначения
plt.xlabel('Плотность'); plt.ylabel('Плотность населения'); plt.title('Плотность населения');

plt.show()

#%%

# создадим новый датафрейм для полиномиальных признаков
poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
# обработаем отуствующие данные
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
poly_target = poly_features['TARGET']
poly_features = poly_features.drop('TARGET', axis=1)
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

# Создадим полиномиальный объект степени 3
poly_transformer = PolynomialFeatures(degree=3)

# Тренировка полиномиальных признаков
poly_transformer.fit(poly_features)

# Трансформация признаков
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)

print('Формат полиномиальных признаков: ', poly_features.shape)

poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]

#%%

# Датафрейм для новых фич
poly_features = pd.DataFrame(poly_features,
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))
# Добавим таргет
poly_features['TARGET'] = poly_target
# рассчитаем корреляцию
poly_corrs = poly_features.corr()['TARGET'].sort_values()
# Отобразим признаки с наивысшей корреляцией
print(poly_corrs.head(10))
print(poly_corrs.tail(5))