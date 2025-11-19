import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
df_train = pd.read_csv("data/perceptron-train.csv", header=None)
df_test = pd.read_csv("data/perceptron-test.csv", header=None)

X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:, 0]

X_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:, 0]

# 2. Обучение без нормализации
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
y_pred_before = clf.predict(X_test)
acc_before = accuracy_score(y_test, y_pred_before)

# 3. Нормализация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # используем параметры из train

# 4. Обучение с нормализацией
clf.fit(X_train_scaled, y_train)
y_pred_after = clf.predict(X_test_scaled)
acc_after = accuracy_score(y_test, y_pred_after)

# 5. Разность
diff = acc_after - acc_before

# Округление до трёх знаков
diff_rounded = round(diff, 3)

print(f"Accuracy до нормализации: {acc_before:.3f}")
print(f"Accuracy после нормализации: {acc_after:.3f}")
print(f"Разность: {diff_rounded}")
