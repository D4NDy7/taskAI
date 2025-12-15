import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# === 1. Загрузка данных ===
data = pd.read_csv("data/data-logistic.csv", header=None)
y = data[0].values          # ответы {-1, 1}
X = data[[1, 2]].values     # два признака


# === 2. Сигмоида ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# === 3. Градиентный спуск ===
def gradient_descent(X, y, k=0.1, C=0, eps=1e-5, max_iter=10000):
    w = np.zeros(X.shape[1])

    for _ in range(max_iter):
        w_old = w.copy()

        # w·x
        margins = y * (X @ w)

        # градиент
        grad = -np.mean((y[:, None] * X) * (1 - sigmoid(margins))[:, None], axis=0)

        # L2-регуляризация
        if C > 0:
            grad += (1 / C) * w

        # шаг
        w -= k * grad

        # проверка сходимости
        if np.linalg.norm(w - w_old) <= eps:
            break

    return w


# === 4. Обучение ===
w_no_reg = gradient_descent(X, y, k=0.1, C=0)
w_reg = gradient_descent(X, y, k=0.1, C=10)


# === 5. Вероятности ===
proba_no_reg = sigmoid(X @ w_no_reg)
proba_reg = sigmoid(X @ w_reg)


# === 6. AUC-ROC ===
auc_no_reg = roc_auc_score(y, proba_no_reg)
auc_reg = roc_auc_score(y, proba_reg)

print("AUC без регуляризации:", auc_no_reg)
print("AUC с регуляризацией:", auc_reg)
