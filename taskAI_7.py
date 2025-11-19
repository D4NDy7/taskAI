from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np

# Загрузка
categories = ["alt.atheism", "sci.space"]
newsgroups = fetch_20newsgroups(
    subset="all", categories=categories, shuffle=False, random_state=241
)
X_text = newsgroups.data
y = newsgroups.target

# TF-IDF только с буквами
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
X = tfidf.fit_transform(X_text)

# Подбор C
C_values = [10**i for i in range(-5, 6)]
svm = LinearSVC(random_state=241, dual=False, max_iter=10000)
kf = KFold(n_splits=5, shuffle=True, random_state=241)

grid = GridSearchCV(svm, {"C": C_values}, cv=kf, scoring="accuracy")
grid.fit(X, y)

best_C = grid.best_params_["C"]
print("Лучший C:", best_C)

# Обучение
final_clf = LinearSVC(C=best_C, random_state=241, dual=False, max_iter=10000)
final_clf.fit(X, y)

# Анализ
coef = final_clf.coef_[0]
feature_names = np.array(tfidf.get_feature_names_out())

# Отладка
print("Число признаков:", len(feature_names))
print("Длина coef:", len(coef))
print("Макс |вес|:", np.max(np.abs(coef)))

# Топ-10
top_indices = np.argsort(np.abs(coef))[-10:]
print("Число индексов:", len(top_indices))
print("Индексы:", top_indices)

top_words = feature_names[top_indices]
top_words_sorted = sorted(top_words)

# Ответ
answer = ",".join(top_words_sorted)
print("Ответ:", answer)
