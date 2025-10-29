from pathlib import Path
import pandas as pd
import numpy as np


# Определяем путь относительно файла
base_dir = Path(__file__).parent
data_path = base_dir / "data" / "titanic.csv"

# Загружаем данные
data = pd.read_csv(data_path, index_col="PassengerId")

# 1. Количество мужчин и женщин
sex_counts = data["Sex"].value_counts()
men = sex_counts.get("male", 0)
women = sex_counts.get("female", 0)
print(men, women)

# 2. Доля выживших (%)
survived_percent = round(data["Survived"].mean() * 100, 2)
print(survived_percent)

# 3. Доля пассажиров 1-го класса (%)
first_class_percent = round((data["Pclass"] == 1).mean() * 100, 2)
print(first_class_percent)

# 4. Средний и медианный возраст
mean_age = round(data["Age"].mean(), 2)
median_age = round(data["Age"].median(), 2)
print(mean_age, median_age)

# 5. Корреляция Пирсона между SibSp и Parch
corr = round(data["SibSp"].corr(data["Parch"]), 2)
print(corr)


# 6. Самое популярное женское имя
def extract_first_name(full_name):
    import re

    match = re.search(r"\(([^)]+)\)", full_name)
    if match:
        name = match.group(1).split()[0]
    else:
        name = full_name.split(",")[1].split(".")[1].strip().split()[0]
    return name


female_names = data[data["Sex"] == "female"]["Name"].apply(extract_first_name)
most_popular_name = female_names.value_counts().idxmax()
print(most_popular_name)
