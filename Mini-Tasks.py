# -*- coding: utf-8 -*-

# -- Mini-Task3 --

!pip install pandas #устанавливаем библиотеки
!pip install pymatgen
!pip install chemparse
!pip install numpy
!pip install sklearn
!pip install scikit-optimize
!pip install shap

import pandas as pd #переносим библиотеки под короткими именами
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Species
import chemparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical 
import shap

db = pd.read_csv('task 3.csv') #записываем базу данных в переменную
constants = pd.read_csv('https://raw.githubusercontent.com/JuliaRJJ/DiZyme_ML/main/constants.csv') #Также прочитаем таблицу с константами

subtype = {'TMB':1.0, 'H2O2':2.0, 'ABTS': 3.0, 'OPD': 4.0, 'DAB':5.0, 'BA':6.0,'TMB+ATP': 1.0, 'H2O2+ATP':2.0, 'TMB*2HCl':1.0, 'AR':7.0}
db['Subtype'].replace(subtype, inplace=True) 
db.Kcat = db['Kcat'].str.replace(" ", "") #убираем пробелы из значений констант каталитической активности

#Добавим новые дескрипторы

#Объём
volume = db['length'] * db['width'] * db['depth']
db = db.assign(volume = volume)
db = db.drop(columns=['length', 'depth', 'width']) #удаляем параметры длины, ширины и высоты

db['formula'] = db['formula'].str.replace('α-', '').str.replace('β-', '').str.replace('y-', '').str.replace('a-', '').str.replace('0.10CeO2', 'Ce0.1O0.2')
#Необходимо немного изменить вид формулы материала, чтобы можно было использовать некоторые функции из pymatgen

'''Данная функция находит средний окислительно-восстановительный потенциал и среднюю плотность заряда на основании данных из 
таблицы констант 
'''
def db_fill(row):  
    rox_sum = 0
    ir_sum = 0
    n = 0
    try:
        elements_list = Composition(chemparse.parse_formula(db.loc[row, 'formula'])).oxi_state_guesses()[0]
    except:
        elements_list, n = [], 1
    else:
        for i in elements_list.keys():
            if i in set(constants.element):
                if elements_list[i] in set(constants.loc[constants.element == i, 'OS']):
                    os = elements_list[i]
                    rox = constants.loc[(constants.element == i) & (constants.OS == os), 'ROx'].mean()
                    ir = constants.loc[(constants.element == i) & (constants.OS == os), 'IR-low'].mean()
                else:
                    os = constants.loc[constants.element == i, 'OS'].mean()
                    rox = constants.loc[constants.element == i, 'ROx'].mean()
                    ir = constants.loc[constants.element == i, 'IR-low'].mean()
                rox_sum += rox
                ir_sum += ir
                n += 1
    finally:
        return(rox_sum/n, ir_sum/n)

#Теперь с помощью предыдущей функции мы можем заполнить все ячейки для окислительно-восстановительного потенциала и плотности заряда
db['Redox'] = [db_fill(x)[0] for x in range(227)]
db['ChargeDensity'] = [db_fill(x)[1] for x in range(227)]

#С использованием pymatgen и chemparse мы можем преобразоать формулу в словарь и затем найти среднюю электроотрицательность
db['electroneg'] = [Composition(chemparse.parse_formula(db.loc[x, 'formula'])).average_electroneg for x in range(227)]

# RFR for Km

y = db.loc[:,'Km'].values #логарифмируем значения констант, чтобы они имени меньший разброс
y = np.log10(y) 
x = db.loc[:,'Syngony':'electroneg'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train) #масштабирование тренировочных данных
x_test = sc.transform(x_test) #масштабирование тестовых данных

regr = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=1, min_samples_split=2, criterion='squared_error', max_features='sqrt')
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
y1_pred = regr.predict(x_train)


Q2 = cross_val_score(regr, x_train, y_train, cv=10)
print('Q2:', Q2.mean())
print('r2_test:', metrics.r2_score(y_test, y_pred))
print('r2_train:', metrics.r2_score(y_train, y1_pred))

# optimization RFR for Km

search_space = {
    "n_estimators": Integer(100,150),
    "criterion": Categorical(['squared_error', 'absolute_error']),
    "min_samples_split": Integer(2, 30),
    "min_samples_leaf": Integer(1, 20),
    "max_depth": Integer(8, 20),
    "max_features": Categorical(['auto', 'sqrt']),
}

opt = BayesSearchCV(estimator = regr, search_spaces=search_space, cv=10, n_iter=32, verbose=2, n_jobs=-1)
opt.fit(x_train, y_train)

n_esti = opt.best_estimator_.n_estimators
criterion = opt.best_estimator_.criterion
min_leaf = opt.best_estimator_.min_samples_leaf
min_split = opt.best_estimator_.min_samples_split
depth = opt.best_estimator_.max_depth
max_feat = opt.best_estimator_.max_features

print('n_esti:', n_esti,'depth:', depth, 'criterion:', criterion, 'min_leaf:', min_leaf,'min_split:', min_split,'max_feat:', max_feat)

# Feature importance RFR for Km
explainer = shap.TreeExplainer(regr)
shap_values = explainer.shap_values(x_test)
f, ax = plt.subplots(figsize=(15, 13))
shap.summary_plot(shap_values, x_test, plot_type='bar')

# RFR for Kcat

y = db.loc[:,'Kcat'].astype(float).values #логарифмируем значения констант, чтобы они имени меньший разброс
y = np.log10(y) 
x = db.loc[:,'Syngony':'electroneg'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train) #масштабирование тренировочных данных
x_test = sc.transform(x_test) #масштабирование тестовых данных

regr = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=1, min_samples_split=2, criterion='squared_error', max_features='sqrt')
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
y1_pred = regr.predict(x_train)


Q2 = cross_val_score(regr, x_train, y_train, cv=10)
print('Q2:', Q2.mean())
print('r2_test:', metrics.r2_score(y_test, y_pred))
print('r2_train:', metrics.r2_score(y_train, y1_pred))

# optimization RFR for Kcat

search_space = {
    "n_estimators": Integer(100,150),
    "criterion": Categorical(['squared_error', 'absolute_error']),
    "min_samples_split": Integer(2, 30),
    "min_samples_leaf": Integer(1, 20),
    "max_depth": Integer(8, 20),
    "max_features": Categorical(['auto', 'sqrt']),
}

opt = BayesSearchCV(estimator = regr, search_spaces=search_space, cv=5, n_iter=32, verbose=2, n_jobs=-1)
opt.fit(x_train, y_train)

n_esti = opt.best_estimator_.n_estimators
criterion = opt.best_estimator_.criterion
min_leaf = opt.best_estimator_.min_samples_leaf
min_split = opt.best_estimator_.min_samples_split
depth = opt.best_estimator_.max_depth
max_feat = opt.best_estimator_.max_features

print('n_esti:', n_esti,'depth:', depth, 'criterion:', criterion, 'min_leaf:', min_leaf,'min_split:', min_split,'max_feat:', max_feat)

# Feature importance RFR for Kcat
explainer = shap.TreeExplainer(regr)
shap_values = explainer.shap_values(x_test)
f, ax = plt.subplots(figsize=(15, 13))
shap.summary_plot(shap_values, x_test, plot_type='bar')

# GBR for Km

y = db.loc[:,'Km'].values #логарифмируем значения констант, чтобы они имени меньший разброс
y = np.log10(y) 
x = db.loc[:,'Syngony':'electroneg'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train) #масштабирование тренировочных данных
x_test = sc.transform(x_test) #масштабирование тестовых данных

regr = GradientBoostingRegressor(learning_rate=0.39, n_estimators=100, max_depth=8, min_samples_leaf=4, min_samples_split=30)
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
y1_pred = regr.predict(x_train)


Q2 = cross_val_score(regr, x_train, y_train, cv=10)
print('Q2:', Q2.mean())
print('r2_test:', metrics.r2_score(y_test, y_pred))
print('r2_train:', metrics.r2_score(y_train, y1_pred))

# optimization GBR for Km

search_space = {
    "n_estimators": Integer(100,150),
    "learning_rate": Real(0.3, 1),
    "min_samples_split": Integer(2, 30),
    "min_samples_leaf": Integer(1, 20),
    "max_depth": Integer(8, 20)
}

opt = BayesSearchCV(estimator = regr, search_spaces=search_space, cv=5, n_iter=32, verbose=2, n_jobs=-1)
opt.fit(x_train, y_train)

n_esti = opt.best_estimator_.n_estimators
l_rate = opt.best_estimator_.learning_rate
min_leaf = opt.best_estimator_.min_samples_leaf
min_split = opt.best_estimator_.min_samples_split
depth = opt.best_estimator_.max_depth
max_feat = opt.best_estimator_.max_features

print('n_esti:', n_esti,'depth:', depth, 'l_rate:', l_rate, 'min_leaf:', min_leaf,'min_split:', min_split)

# Feature importance GBR for Km
explainer = shap.TreeExplainer(regr)
shap_values = explainer.shap_values(x_test)
f, ax = plt.subplots(figsize=(15, 13))
shap.summary_plot(shap_values, x_test, plot_type='bar')

# GBR for Kcat

y = db.loc[:,'Kcat'].astype(float).values #логарифмируем значения констант, чтобы они имени меньший разброс
y = np.log10(y) 
x = db.loc[:,'Syngony':'electroneg'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train) #масштабирование тренировочных данных
x_test = sc.transform(x_test) #масштабирование тестовых данных

regr = GradientBoostingRegressor(learning_rate=0.39, n_estimators=100, max_depth=8, min_samples_leaf=4, min_samples_split=30)
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
y1_pred = regr.predict(x_train)


Q2 = cross_val_score(regr, x_train, y_train, cv=10)
print('Q2:', Q2.mean())
print('r2_test:', metrics.r2_score(y_test, y_pred))
print('r2_train:', metrics.r2_score(y_train, y1_pred))

# optimization GBR for Kcat

search_space = {
    "n_estimators": Integer(100,150),
    "learning_rate": Real(0.3, 1),
    "min_samples_split": Integer(2, 30),
    "min_samples_leaf": Integer(1, 20),
    "max_depth": Integer(8, 20)
}

opt = BayesSearchCV(estimator = regr, search_spaces=search_space, cv=5, n_iter=32, verbose=2, n_jobs=-1)
opt.fit(x_train, y_train)

n_esti = opt.best_estimator_.n_estimators
l_rate = opt.best_estimator_.learning_rate
min_leaf = opt.best_estimator_.min_samples_leaf
min_split = opt.best_estimator_.min_samples_split
depth = opt.best_estimator_.max_depth
max_feat = opt.best_estimator_.max_features

print('n_esti:', n_esti,'depth:', depth, 'l_rate:', l_rate, 'min_leaf:', min_leaf,'min_split:', min_split)

# Feature importance GBR for Kcat
explainer = shap.TreeExplainer(regr)
shap_values = explainer.shap_values(x_test)
f, ax = plt.subplots(figsize=(15, 13))
shap.summary_plot(shap_values, x_test, plot_type='bar')

# EN for Km

y = db.loc[:,'Km'].values #логарифмируем значения констант, чтобы они имени меньший разброс
y = np.log10(y) 
x = db.loc[:,'Syngony':'electroneg'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train) #масштабирование тренировочных данных
x_test = sc.transform(x_test) #масштабирование тестовых данных

regr = ElasticNet(alpha=0.005, l1_ratio=0.68, max_iter=1100, selection='cyclic')
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
y1_pred = regr.predict(x_train)


Q2 = cross_val_score(regr, x_train, y_train, cv=10)
print('Q2:', Q2.mean())
print('r2_test:', metrics.r2_score(y_test, y_pred))
print('r2_train:', metrics.r2_score(y_train, y1_pred))

# optimization EN for Km

search_space = {
    "alpha": Real(0.001, 1.0),
    "l1_ratio": Real(0.001, 1.0),
    "max_iter": Integer(900, 1200),
    "selection": Categorical(['cyclic', 'random'])
}

opt = BayesSearchCV(estimator = regr, search_spaces=search_space, cv=3, n_iter=32, verbose=2, n_jobs=-1)
opt.fit(x_train, y_train)

alpha = opt.best_estimator_.alpha
l1_ratio = opt.best_estimator_.l1_ratio
max_iter = opt.best_estimator_.max_iter
selection = opt.best_estimator_.selection

print('alpha:', alpha, 'l1_ratio:', l1_ratio, 'max_iter:', max_iter, 'selection:', selection)

# Feature importance EN for Km

feature_importance = pd.Series(data = np.abs(regr.coef_))

n_selected_features = (feature_importance>0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (18,6))

# EN for Kcat

y = db.loc[:,'Kcat'].astype(float).values #логарифмируем значения констант, чтобы они имени меньший разброс
y = np.log10(y) 
x = db.loc[:,'Syngony':'electroneg'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train) #масштабирование тренировочных данных
x_test = sc.transform(x_test) #масштабирование тестовых данных

regr = ElasticNet(alpha=0.0159, l1_ratio=0.0001, max_iter=826, selection='random')
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
y1_pred = regr.predict(x_train)


Q2 = cross_val_score(regr, x_train, y_train, cv=10)
print('Q2:', Q2.mean())
print('r2_test:', metrics.r2_score(y_test, y_pred))
print('r2_train:', metrics.r2_score(y_train, y1_pred))

# optimization EN for Kcat

search_space = {
    "alpha": Real(0.0001, 1.0),
    "l1_ratio": Real(0.0001, 1.0),
    "max_iter": Integer(800, 1200),
    "selection": Categorical(['cyclic', 'random'])
}

opt = BayesSearchCV(estimator = regr, search_spaces=search_space, cv=3, n_iter=32, verbose=2, n_jobs=-1)
opt.fit(x_train, y_train)

alpha = opt.best_estimator_.alpha
l1_ratio = opt.best_estimator_.l1_ratio
max_iter = opt.best_estimator_.max_iter
selection = opt.best_estimator_.selection

print('alpha:', alpha, 'l1_ratio:', l1_ratio, 'max_iter:', max_iter, 'selection:', selection)

# Feature importance EN for Kcat

feature_importance = pd.Series(data = np.abs(regr.coef_))

n_selected_features = (feature_importance>0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))

feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (18,6))

# -- Mini-Task 1 --

#Устанавливаем все библиотеки которые нам потом понадобятся
!pip install pandas
!pip install seaborn
!pip install sklearn
!pip install numpy
!pip install matplotlib
!pip install openpyxl

#Импортируем библиотеки
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

#Читаем две базы данных
db1 = pd.read_excel('Database_1.xlsx')
db2 = pd.read_excel('Database_2.xlsx')

#Также прочитаем второй лист одной из баз данных, на котором находятся некоторые параметры для наночастиц
db2_sheet2 = pd.read_excel('Database_2.xlsx', sheet_name = "Лист2")
#Разделим данный датафрейм на два, один для оксидов, другой для элементов, а также переименуем некоторые колонки
db2_sheet2_o = db2_sheet2[:17].rename(columns={'Material type': 'NP_type', 'Topological polar surface area (Å²)': 'surface_area', 'Electronegativity': 'electronegativity'})
db2_sheet2_e = db2_sheet2[38:].rename(columns={'Material type': 'NP_type', 'Elements': 'electronegativity'}).loc[:, ["NP_type", "electronegativity"]]

'''
Данная функция заполняет значение в одной колонке на основании значения в другой колонке
Она принимает на вход датафрейм, колонку, в которой есть пустое или неверное значение,
колонку, на основании которой можно заполнить пустое значение из другой колонки, а также индекс такого значения
Таким образом мы заполняем неверное значение из первой колонки таким образом, чтобы оно было равно наиболее частому значению 
из этой колонки, которому соответсвовало бы то же значение из второй колонки
Пример для Database_2: Если у нас пропущено какое-то значение в графе Material type то мы можем посмотреть в графе Elements 
какой элемент соответсвует данной частице и из него понять что это была за частица, сравнив с другими частицами 
с тем же элементом
'''
def fill_np(db, column1, column2, index1):
    db.loc[index1, column1] = db[db[column2] == db.loc[index1, column2]][column1] \
                              .value_counts().idxmax()

'''Данная функция заполняет пустые значения названий наночастиц, а также неподходящие названия (wrong_values)
в базе данных на основе тех значений которые соответсвуют значениям из другой колонки (reference_column)
Также данная функция переименовывает данные колонки в базах данных
'''
def np_imputer(db, old_name, wrong_values, reference_column):
    db = db.rename(columns={old_name: 'NP_type'})
    values_to_fill = db[db['NP_type'].isnull()].index.tolist()
    for i in list(set(db['NP_type']).intersection(wrong_values)):
        values_to_fill += db[db['NP_type'] == i].index.tolist()
    for i in values_to_fill:
        fill_np(db, 'NP_type', reference_column, i)
    return db

#Теперь можно применить описанную выше функцию для обоих баз данных
db2 = np_imputer(db2, 'Material type', ["don't remember"], 'Elements')
db1 = np_imputer(db1, 'Nanoparticle', ["Copper Oxide", 'Zinc oxide'], 'Particle ID')

#Также нам понадобится функция, которая будет заполнять значения в базах данных на основе второго листа второй базы данных
def db_filler(db, column_name, db2):
    db[column_name] = [db2.loc[db2.NP_type == x][column_name].values[0] \
    if x in set(db2['NP_type']) else db.loc[db.NP_type == x][column_name].values[0] for x in db['NP_type']]

'''Мы можем использовать лист 2 во второй базе данных для того, чтобы заполнить некоторые значения в обоих базах данных
Для начала попробуем восстановить значение Topological polar surface area (Å²) во второй базе данных'''
#Сначала переименуем колонку в более лаконичное название
db2 = db2.rename(columns={'Topological polar surface area (Å²)': 'surface_area'})
#Теперь заполним колонку значениями из второго листа
db2['surface_area'] = [np.nan for x in db2['NP_type']]
db_filler(db2, 'surface_area', db2_sheet2_o)
#Теперь создадим такую же колонку в первой базе данных и попробуем заполнить ее аналогичным образом
db1['surface_area'] = [np.nan for x in db1['NP_type']]
db_filler(db1, 'surface_area', db2_sheet2_o)

#Теперь мы можем посмотреть на графиках насколько хорошо мы выполнили заполнение
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(26,16))

db1_surface_plot = db1['surface_area'].value_counts(dropna = False)
db2_surface_plot = db2['surface_area'].value_counts(dropna = False)

my_grid = axes.flatten()
db1_surface_plot.plot(kind='bar', ylabel='db1', xlabel='Value', ax=my_grid[0])
db2_surface_plot.plot(kind='bar', ylabel='db2', xlabel='Value', ax=my_grid[1])

'''Исходя из двух данных графиков мы видим, что более половины значений остаются неизвестными даже после заполнения, так что 
вероятно ни один другой способ заполнения пропущенных значений не будет достаточно адекватным
Таким образом следующие значения из базы данных можно убрать так как мы все равно их не сможем заполнить:
- Topological polar surface area (Å²)
- a (Å)
- b (Å)
- c (Å)
- α (°)
- β (°)
- γ (°)
- Density (g/cm3)
'''

#Теперь попробуем заполнить колонку Electronegativity, так как для нее имеется чуть больше данных
db2 = db2.rename(columns={'Electronegativity': 'electronegativity'})
db2['electronegativity'] = [np.nan for x in db2['NP_type']]
db_filler(db2, 'electronegativity', db2_sheet2_o)
db_filler(db2, 'electronegativity', db2_sheet2_e)
#Теперь создадим такую же колонку в первой базе данных и попробуем заполнить ее аналогичным образом
db1['electronegativity'] = [np.nan for x in db1['NP_type']]
db_filler(db1, 'electronegativity', db2_sheet2_o)
db_filler(db1, 'electronegativity', db2_sheet2_e)

#Теперь мы можем посмотреть на графиках насколько хорошо мы выполнили заполнение
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(26,16))

db1_electronegativity_plot = db1['electronegativity'].value_counts(dropna = False)
db2_electronegativity_plot = db2['electronegativity'].value_counts(dropna = False)

my_grid = axes.flatten()
db1_electronegativity_plot.plot(kind='bar', ylabel='db1', xlabel='Value', ax=my_grid[0])
db2_electronegativity_plot.plot(kind='bar', ylabel='db2', xlabel='Value', ax=my_grid[1])

'''Мы видим, что в случае электроотрицательности пустых значений осталось меньше, так что пока можно оставить...
Теперь добавим во вторую базу данных тип частиц, так как в ней все частицы неорганические'''
db2['inorganic'] = ['I' for x in db2['NP_type']]
#Также переведем концентрацию в μM
db2['doze_μM'] = db2['Exposure dose (ug/mL)'] / db2['Molecular weight (g/mol)'] / 1000

'''Теперь посмотрим какие значения оставлять, а какие нет, опираясь на следующие графики, которые показывают частоту,
с которой встречается каждое значение для ряда колонок'''

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(32,16))
v = db1['Cells'].value_counts(dropna = False)
cell_plot = db1[db1['Cells'].isin(v.index[v.gt(25)])]['Cells'].value_counts(dropna = False)
animal_plot = db1['Animal?'].value_counts(dropna = False)
coat_plot = db1['coat'].value_counts(dropna = False)
morph_plot = db1['Cell morphology'].value_counts(dropna = False)
pos_plot = db1['Positive control (Y/N)'].value_counts(dropna = False)
stability_plot = db1['Colloidal stability checked (Y/N)'].value_counts(dropna = False)
interference_plot = db1['Interference checked (Y/N)'].value_counts(dropna = False)
test_plot = db1['Test'].value_counts(dropna = False)
biochemical_plot = db1['Biochemical metric'].value_counts(dropna = False)
indicator_plot = db1['Test indicator'].value_counts(dropna = False)

my_grid = axes.flatten()
cell_plot.plot(kind='bar', ylabel='Cells', xlabel='Value', ax=my_grid[0])
animal_plot.plot(kind='bar', ylabel='Animal', xlabel='Value', ax=my_grid[1])
coat_plot.plot(kind='bar', ylabel='Coat', xlabel='Value', ax=my_grid[2])
morph_plot.plot(kind='bar', ylabel='Morphology', xlabel='Value', ax=my_grid[3])
pos_plot.plot(kind='bar', ylabel='Positive control', xlabel='Value', ax=my_grid[4])
stability_plot.plot(kind='bar', ylabel='Colloidal stability', xlabel='Value', ax=my_grid[5])
interference_plot.plot(kind='bar', ylabel='Interference', xlabel='Value', ax=my_grid[6])
test_plot.plot(kind='bar', ylabel='Test', xlabel='Value', ax=my_grid[7])
biochemical_plot.plot(kind='bar', ylabel='Biochemical metric', xlabel='Value', ax=my_grid[8])
indicator_plot.plot(kind='bar', ylabel='Test indicator', xlabel='Value', ax=my_grid[9])

'''
Соответствие старых названий и новых
NP_type         - Material type из db2 и Nanoparticle из db1
viability_%     - Viability (%) из db2 и % Cell viability из db1
charge_mV       - Surface charge (mV) из db2 и Zeta potential (mV) из db1
size_nm         - Hydro size (nm) из db2 и Diameter (nm) из db1
doze_μM         - Exposure dose (ug/mL) из db2 и Concentration μM из db1

Ненужные столбцы, которые надо удалить
db1:
Reference DOI       - Характеристика, которая не имеет отношения к эксперименту
Publication year    - Характеристика, которая не имеет отношения к эксперименту
Particle ID         - Не нужна после того, как была использована для заполнения пропущенных частиц
Cells               - db1['Cells'].unique() показывает, что данный параметр может принимать 81 разное значение и этого параметра нет во второй базе данных
coat                - Аналогичная ситуация, большая часть частиц не имеет этого параметра
Animal?             - Аналогичная ситуация, большая часть частиц не имеет этого параметра
Cell morphology     - Тоже слишком много значений и нету во второй базе данных

Все параметры ниже имеют похожие проблемы
Positive control (Y/N)
Colloidal stability checked (Y/N)
Interference checked (Y/N)
Test
Test indicator
Biochemical metric
Cell-organ/tissue source
Cell age: embryonic (E), Adult (A)
Cell line (L)/primary cells (P)
Human(H)/Animal(A) cells
Exposure time (h)

db2:
Elements
Cell type
Ionic radius
Core size (nm)
Surface area (m2/g)
Number of atoms
Molecular weight (g/mol)
и тд
'''

#Теперь удалим ненужные колонки

db1 = db1.drop(columns=['Reference DOI', 'Publication year', 'Particle ID', 'coat', 'Animal?', 'Cell morphology',
                        'Positive control (Y/N)', 'Colloidal stability checked (Y/N)', 'Interference checked (Y/N)',
                        'Test', 'Test indicator', 'Biochemical metric', 'Cell-organ/tissue source', 'Cell age: embryonic (E), Adult (A)',
                        'Cell line (L)/primary cells (P)', 'Human(H)/Animal(A) cells', 'Exposure time (h)', 'surface_area', 'Cells'])
db2 = db2.drop(columns=['Elements', 'Ionic radius', 'Core size (nm)', 'Surface area (m2/g)', 'Number of atoms', 
                        'Molecular weight (g/mol)', 'surface_area', 'a (Å)', 'b (Å)', 'c (Å)', 'α (°)', 'β (°)', 
                        'γ (°)', 'Density (g/cm3)', 'Cell type','Exposure dose (ug/mL)'])

#Теперь переименуем колонки и объедим базы данных
db1.columns = ['NP_type', 'inorganic', 'size_nm', 'doze_μM', 'charge_mV', 'viability_%', 'electronegativity']
db2.columns = ['NP_type', 'electronegativity', 'size_nm', 'charge_mV', 'viability_%', 'inorganic', 'doze_μM']

final_db = pd.concat([db1, db2], ignore_index=True, sort=False)

#Сделаем так, чтобы для параметра viability_% были только положительные значения, не превышающие 100%
final_db['viability_%'] = final_db['viability_%'].abs()
#final_db.loc[final_db['viability_%'] > 100, 'viability_%'] = np.nan

#Также исправим опечатку в колонке inorganic
final_db.inorganic = final_db.inorganic.replace('O', 0)
final_db.inorganic = final_db.inorganic.replace('I', 1)

#строим Boxplot для того, чтобы понять есть ли в наших данных какие-то выбросы
y = final_db.loc[:,'viability_%'].values
y1 = final_db.loc[:,'charge_mV'].values 
y2 = np.log10(final_db.loc[:,'doze_μM'].values) 
y3 = np.log10(final_db.loc[:,'size_nm'].values)
y4 = final_db.loc[:,'electronegativity'].values

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(30,20))
my_grid = axes.flatten()

plot_1 = sb.boxplot(x=final_db['inorganic'], y=y, palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[0])
#plot_1 = sb.stripplot(x=final_db['inorganic'], y=y, palette=['#615289', '#678476'], dodge=True, ax=my_grid[0])

plot_2 = sb.boxplot(x=final_db['inorganic'], y=y1, palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[1])
#plot_2 = sb.stripplot(x=final_db['inorganic'], y=y1, palette=['#615289', '#678476'], dodge=True, ax=my_grid[1])

plot_3 = sb.boxplot(x=final_db['inorganic'], y=y2, palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[2])
#plot_3 = sb.stripplot(x=final_db['inorganic'], y=y2, palette=['#615289', '#678476'], dodge=True, ax=my_grid[2])

plot_4 = sb.boxplot(x=final_db['inorganic'], y=y3, palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[3])
#plot_3 = sb.stripplot(x=final_db['inorganic'], y=y2, palette=['#615289', '#678476'], dodge=True, ax=my_grid[2])

plot_5 = sb.boxplot(x=final_db['inorganic'], y=y4, palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[4])
#plot_3 = sb.stripplot(x=final_db['inorganic'], y=y2, palette=['#615289', '#678476'], dodge=True, ax=my_grid[2])

#Уберем выбросы для колонки charge_mV и doze_μM
q = final_db['charge_mV'].quantile(0.95)
final_db[final_db['charge_mV'] > q] = np.nan

q = final_db['doze_μM'].quantile(0.99)
final_db[final_db['doze_μM'] > q] = np.nan

#используем метод kNN для заполнения недостающих данных
imputer = KNNImputer(n_neighbors=5)  #введем функцию для метода kNN с использованием для предсказания 5 ближайших соседей
db_filled = imputer.fit_transform(final_db.loc[:,'inorganic':]) #записываем в новую переменную заполненную базу данных
impute_data = pd.DataFrame(imputer.fit_transform(db_filled), columns=final_db.loc[:,'inorganic':].columns)

impute_data['NP_type'] = final_db['NP_type']
final_db = impute_data

final_db.inorganic = final_db.inorganic.replace('I', 1)

#Заменим неверное значение в inorganic
final_db.loc[final_db['inorganic'] > 0, 'inorganic'] = 1

final_db

# -- Mini-Task2 --

'''
Нами был выбран следующий порядок выполнения мини-таска:
1. Обработка данных: удаление лишних строк и столбцов, перевод данных в числовой формат
2. Заполнение пропусков методом kNN
3. Boxplot
4. PCA
5. Построение корреляционной матрицы из числовых и категориальных параметров
6. Предложение кандидатов на выброс
'''

!pip install pandas #устанавливаем библиотеки
!pip install seaborn
!pip install sklearn
!pip install numpy
!pip install matplotlib

import pandas as pd #переносим библиотеки под короткими именами
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

db = pd.read_csv('DiZyme_jan2022.csv') #записываем базу данных в переменную
db

#переводим качественные характеристики в числовые, при этом используя inplace=True перезаписываем данные
subtype = {'TMB':1.0, 'H2O2':2.0, 'ABTS': 3.0, 'OPD': 4.0, 'DAB':5.0, 'BA':6.0,'TMB+ATP': 1.0, 'H2O2+ATP':2.0, 'TMB*2HCl':1.0, 'AR':7.0}
db['Subtype'].replace(subtype, inplace=True) 
activity = {'peroxidase':1.0, 'oxidase':2.0, 'catalase': 3.0}
db['activity'].replace(activity, inplace=True)
db.loc[(db.surface != 'naked'), 'surface'] = 1.0 #так как в столбце surface 22 различных параметра, при этом naked преобладает, имеет смысл выделить только naked и остальные
db.loc[(db.surface == 'naked'), 'surface'] = 0.0
db.Kcat = db['Kcat'].str.replace(" ", "") #убираем пробелы из значений констант каталитической активности
db

db = db.drop(columns=['link', 'formula']) #отбрасываем неинтерпретируемые столбцы 
db = db[:303] #отбрасываем лишние строки
db

#используем метод kNN для заполнения недостающих данных
imputer = KNNImputer(n_neighbors=5)  #введем функцию для метода kNN с использованием для предсказания 5 ближайших соседей
db_filled = imputer.fit_transform(db) #записываем в новую переменную заполненную базу данных
impute_data = pd.DataFrame(imputer.fit_transform(db_filled), columns=db.columns)
db = impute_data
db

#строим Boxplot 
y = db.loc[:,'Km'].values #в качестве оси y выберем значения константы Михаэлиса-Ментен и y1 константы каталитической активности
y = np.log10(y) #прологорифмируем значения, чтобы они имени меньший разброс
y1 = db.loc[:,'Kcat'].values 
y1 = np.log10(y1)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30,20)) #создаем место для 4-х графиков
my_grid = axes.flatten()

plot_1 = sb.boxplot(x=db['activity'], y=y, hue=db['surface'], palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[0]) #строим боксплот в 1 ячейке
plot_1 = sb.swarmplot(x=db['activity'], y=y, hue=db['surface'], palette=['#615289', '#678476'], dodge=True, ax=my_grid[0]) #там же строим точки для большей наглядности
plot_1.set_ylabel( r'$\log_{10}Km, s^{-1}$', fontsize = 15) #обозначаем ось y
plot_1.set_xlabel( 'Catalitic activity', fontsize = 15) #ось y
plot_1.set_title('Boxplot of coated and naked nanozymes with peroxidase, oxidase and catalase activity as a function of lgKm') #заголовок графика

plot_2 = sb.boxplot(x=db['Dstr'], y=y, hue=db['surface'], palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[1])
plot_2 = sb.swarmplot(x=db['Dstr'], y=y, hue=db['surface'], palette=['#615289', '#678476'], dodge=True, ax=my_grid[1])
plot_2.set_ylabel( r'$\log_{10}Km, s^{-1}$', fontsize = 15)
plot_2.set_xlabel( 'Dimensionality', fontsize = 15)
plot_2.set_title('Boxplot of coated and naked nanozymes with 1d, 2d and 3d dimensionality as a function of lgKm')

plot_3 = sb.boxplot(x=db['activity'], y=y1, hue=db['surface'], palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[2])
plot_3 = sb.swarmplot(x=db['activity'], y=y1, hue=db['surface'], palette=['#615289', '#678476'], dodge=True, ax=my_grid[2])
plot_3.set_ylabel( r'$\log_{10}Kcat, s^{-1}$', fontsize = 15)
plot_3.set_xlabel( 'Catalitic activity', fontsize = 15)
plot_3.set_title('Boxplot of coated and naked nanozymes with peroxidase, oxidase and catalase activity as a function of lgKcat')

plot_4 = sb.boxplot(x=db['Dstr'], y=y1, hue=db['surface'], palette=['#9C7AE7', '#8FC1AB'], ax=my_grid[3])
plot_4 = sb.swarmplot(x=db['Dstr'], y=y1, hue=db['surface'], palette=['#615289', '#678476'], dodge=True, ax=my_grid[3])
plot_4.set_ylabel( r'$\log_{10}Kcat, s^{-1}$', fontsize = 15)
plot_4.set_xlabel( 'Dimensionality', fontsize = 15)
plot_4.set_title('Boxplot of coated and naked nanozymes with 1d, 2d and 3d dimensionality as a function of lgKcat')


#PCA
from matplotlib.pyplot import figure

x = db.drop(columns=['Km','Vmax','Kcat']).values #отбираем столбцы, которые будем использовать
scaler = MinMaxScaler(feature_range=(0, 1)) 
x = scaler.fit_transform(X) #применяем масштабирование функций в данных
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

y = db.loc[:,'Km'].values #логарифмируем значения констант, чтобы они имени меньший разброс
y = np.log10(y) 
figure(figsize=(10, 6)) #задаем размер изображения
plot_4 = plt.scatter(principalComponents[:303, 0], principalComponents[:303, 1], c = y[:303]) #строим график
label = 'Km' #подписываем график
plt.colorbar(label=label) #задаем на графике цветовую шкалу

y1 = db.loc[:,'Kcat'].values
y1 = np.log10(y1)
figure(figsize=(10, 6))
plot_5 = plt.scatter(principalComponents[:303, 0], principalComponents[:303, 1], c = y1[:303])
label = 'Kcat'
plt.colorbar(label=label)

y2 = db.loc[:,'mCD'].values
figure(figsize=(10, 6))
plot_5 = plt.scatter(principalComponents[:303, 0], principalComponents[:303, 1], c = y2[:303])
label = 'mCD'
plt.colorbar(label=label)

y3 = db.loc[:,'mROx'].values
figure(figsize=(10, 6))
plot_6 = plt.scatter(principalComponents[:303, 0], principalComponents[:303, 1], c = y3[:303])
label = 'mROx'
plt.colorbar(label=label)

y4 = db.loc[:,'temp'].values
figure(figsize=(10, 6))
plot_7 = plt.scatter(principalComponents[:303, 0], principalComponents[:303, 1], c = y4[:303])
label = 'temp'
plt.colorbar(label=label)

mask = np.triu(np.ones_like(db.corr(), dtype=bool)) #создаем маску для выделения верхнего треугольника массива
cmap = sb.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True) #создание палитры для корреляционного графика
plt.figure(figsize=(17, 14)) #создание графика и определение его размеров
plot = sb.heatmap(db.corr(), annot=True, fmt=".2f", mask=mask, vmin=-1, vmax=1, center= 0, cmap=cmap, linewidth=0.0005) 
#annot=True - вывод коэффициентов корреляции, fmt - число знаков после запятой, v - пределы, cmap - палитра, linewidth - толщина шрифта

#удаляем параметры длины, ширины и высоты, так как они имеют высокую корреляцию между собой и с объёмом
db = db.drop(columns=['length', 'depth', 'width'])
db

mask = np.triu(np.ones_like(db.corr(), dtype=bool)) #создаем маску для выделения верхнего треугольника массива
cmap = sb.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True) #создание палитры для корреляционного графика
plt.figure(figsize=(17, 14)) #создание графика и определение его размеров
plot = sb.heatmap(db.corr(), annot=True, fmt=".2f", mask=mask, vmin=-1, vmax=1, center= 0, cmap=cmap, linewidth=0.0005) 
#annot=True - вывод коэффициентов корреляции, fmt - число знаков после запятой, v - пределы, cmap - палитра, linewidth - толщина шрифта

