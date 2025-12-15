import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
import numpy as np

# === 1. Загрузка и предобработка трейна ===
train = pd.read_csv('data/salary-train.csv')

def text_clean(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

train['FullDescription'] = train['FullDescription'].fillna('').apply(text_clean)
train['LocationNormalized'] = train['LocationNormalized'].fillna('nan')
train['ContractTime'] = train['ContractTime'].fillna('nan')

# === 2. TF-IDF для описаний ===
tfidf = TfidfVectorizer(min_df=5)
X_text = tfidf.fit_transform(train['FullDescription'])

# === 3. One-hot для категориальных признаков ===
enc = DictVectorizer()
X_cat = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

# === 4. Объединение признаков ===
X_train = hstack([X_text, X_cat])
y_train = train['SalaryNormalized']

# === 5. Обучение Ridge ===
model = Ridge(alpha=1, random_state=241)
model.fit(X_train, y_train)

# === 6. Предобработка теста ===
test = pd.read_csv('data/salary-test-mini.csv')
test['FullDescription'] = test['FullDescription'].fillna('').apply(text_clean)
test['LocationNormalized'] = test['LocationNormalized'].fillna('nan')
test['ContractTime'] = test['ContractTime'].fillna('nan')

X_test_text = tfidf.transform(test['FullDescription'])
X_test_cat = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_cat])

# === 7. Прогноз ===
preds = model.predict(X_test)
result = ' '.join([f"{p:.2f}" for p in preds])
print(result)