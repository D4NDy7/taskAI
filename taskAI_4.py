import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale


# 1. Загружаем housing.csv
df = pd.read_csv("data/housing.csv")

print("Форма датасета:", df.shape)
print(df.head())

# 2. X и y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 3. Масштабирование
X_scaled = scale(X)

# 4. Сетка p
p_values = np.linspace(1, 10, 200)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_score = -1e18
best_p = None

for p in p_values:
    model = KNeighborsRegressor(
        n_neighbors=5, weights="distance", metric="minkowski", p=p
    )
    scores = cross_val_score(
        model, X_scaled, y, cv=kf, scoring="neg_mean_squared_error"
    )
    mean_score = scores.mean()

    if mean_score > best_score:
        best_score = mean_score
        best_p = p

print("Лучшее p =", best_p.round(2))
print("Лучшая средняя ошибка =", best_score.round(2))
