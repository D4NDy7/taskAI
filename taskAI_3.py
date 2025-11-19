import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

# 1. Загружаем данные
df = pd.read_csv("data/wine.data", header=None)

# Классы — первый столбец
y = df[0]

# Признаки — столбцы 1..13
X = df.loc[:, 1:]

# 2. Создаём KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Проверяем k от 1 до 50
accuracies = []
for k in range(1, 51):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
    accuracies.append(scores.mean())

best_k_before = np.argmax(accuracies) + 1
best_score_before = accuracies[best_k_before - 1]

print("Без масштабирования:")
print("Лучшее k =", best_k_before)
print("Accuracy =", best_score_before)

# 4. Масштабируем признаки
X_scaled = scale(X)

accuracies_scaled = []
for k in range(1, 51):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="accuracy")
    accuracies_scaled.append(scores.mean())

best_k_after = np.argmax(accuracies_scaled) + 1
best_score_after = accuracies_scaled[best_k_after - 1]

print("\nПосле масштабирования:")
print("Лучшее k =", best_k_after)
print("Accuracy =", best_score_after)
