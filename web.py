import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Загрузка данных
data = pd.read_csv('housing.csv')

mean_total_bedrooms = data['total_bedrooms'].mean()
data['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)
# Создание словаря для соответствия категорий и чисел
categories_to_numbers = {
    '<1H OCEAN': 1,
    'INLAND': 2,
    'NEAR OCEAN': 3,
    'NEAR BAY': 4,
    'ISLAND': 5
}

# Замена категорий числами в столбце 'ocean_proximity'
data['ocean_proximity'] = data['ocean_proximity'].map(categories_to_numbers)

# Разделение данных на признаки и целевую переменную
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Создание интерфейса для изменения гиперпараметров KNN
st.title("K-Nearest Neighbors")

st.sidebar.header('Изменение гиперпараметров KNN:')
n_neighbors = st.sidebar.slider("Количество соседей (n_neighbors):", 1, 20, 5)
weights = st.sidebar.selectbox("Веса (weights):", ('uniform', 'distance'))

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели KNN
model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
model.fit(X_train, y_train)

# Предсказание и оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)


st.write("Accuracy:", accuracy)

#График зависимости точности от количества соседей:
accuracy_scores = []
neighbors_range = range(1, 21)
for n in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(neighbors_range, accuracy_scores, marker='o')
plt.title('Точность модели KNN в зависимости от количества соседей')
plt.xlabel('Количество соседей')
plt.ylabel('Точность')
st.pyplot(plt)

#График зависимости точности от типа весов::
weight_types = ['uniform', 'distance']
accuracy_scores = []
for w in weight_types:
    knn = KNeighborsClassifier(weights=w)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.bar(weight_types, accuracy_scores)
plt.title('Точность модели KNN в зависимости от типа весов')
plt.xlabel('Тип весов')
plt.ylabel('Точность')
st.pyplot(plt)

#Кривая обучения:
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
    X, y, cv=5, scoring='accuracy', n_jobs=-1, 
    train_sizes=np.linspace(0.01, 1.0, 50))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.title('Кривая обучения для модели KNN')
plt.xlabel('Размер обучающего набора')
plt.ylabel('Точность')
plt.legend(loc='best')
st.pyplot(plt)