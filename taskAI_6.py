import pandas as pd
from sklearn.svm import SVC

# 1. Загрузка данных
df = pd.read_csv("data/svm-data.csv", header=None)
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# 2. Обучение SVM
clf = SVC(kernel="linear", C=100000, random_state=241)
clf.fit(X, y)

# 3. Получение индексов опорных векторов (0-based → 1-based)
support_indices = clf.support_ + 1  # перевод в нумерацию с 1

# 4. Сортировка (на всякий случай)
support_indices_sorted = sorted(support_indices)

# 5. Форматирование ответа
answer = ",".join(map(str, support_indices_sorted))

print(answer)
