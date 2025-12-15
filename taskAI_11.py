import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 1. Загрузка
close_prices = pd.read_csv('data/close_prices.csv')
djia_prices = pd.read_csv('data/djia_index.csv')

X = close_prices.iloc[:, 1:]
y_djia = djia_prices.iloc[:, 1].values

# 2. PCA
pca = PCA(n_components=10)
pca.fit(X)

# Сколько компонент для 90% дисперсии?
n_for_90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1
print("1. Компонент для 90% дисперсии:", n_for_90)

# 3. Первая компонента
first_comp = pca.transform(X)[:, 0]

# 4. Корреляция с DJIA
corr = np.corrcoef(first_comp, y_djia)[0, 1]
print("2. Корреляция с индексом Доу-Джонса:", f"{corr:.2f}")

# 5. Компания с максимальным вкладом
company_names = X.columns
weights_first = pca.components_[0]
best_company = company_names[np.argmax(np.abs(weights_first))]
print("3. Компания с наибольшим весом:", best_company)