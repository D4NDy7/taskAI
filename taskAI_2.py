from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# Определяем путь относительно файла
base_dir = Path(__file__).parent
data_path = base_dir / "data" / "titanic.csv"

# Загружаем данные
data = pd.read_csv(data_path, index_col="PassengerId")

# Выбираем признаки
features = ["Pclass", "Fare", "Age", "Sex"]
X = data[features]
y = data["Survived"]

# Преобразуем пол в числовое значение
X = data[["Pclass", "Fare", "Age", "Sex"]].copy()
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# Удаляем строки с пропущенными данными
X = X.dropna()
y = y[X.index]

# Обучаем модель
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

# Получаем важности признаков
importances = pd.Series(clf.feature_importances_, index=features)
top_features = importances.sort_values(ascending=False).head(2)
print(top_features.index.tolist())
