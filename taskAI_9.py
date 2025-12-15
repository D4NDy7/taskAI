import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

# === Часть 1 и 2: Анализ classification.csv ===
df = pd.read_csv('data/classification.csv')
y_true = df['true']
y_pred = df['pred']

# Confusion matrix компоненты
TP = ((y_true == 1) & (y_pred == 1)).sum()
FP = ((y_true == 0) & (y_pred == 1)).sum()
FN = ((y_true == 1) & (y_pred == 0)).sum()
TN = ((y_true == 0) & (y_pred == 0)).sum()

# Метрики качества
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("TP FP FN TN:")
print(TP, FP, FN, TN)

print("\nAccuracy Precision Recall F1:")
print(f"{acc:.2f} {prec:.2f} {rec:.2f} {f1:.2f}")

# === Часть 3–6: Анализ scores.csv ===
scores_df = pd.read_csv('data/scores.csv')
y_true_scores = scores_df['true']

classifiers = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']

# Часть 3: AUC-ROC
aucs = {}
for clf in classifiers:
    aucs[clf] = roc_auc_score(y_true_scores, scores_df[clf])

best_by_auc = max(aucs, key=aucs.get)
print("\nЛучший по AUC-ROC:")
print(best_by_auc)

# Часть 4–6: Precision при Recall >= 0.7
best_precision = -1
best_clf_for_prec = ""

for clf in classifiers:
    prec_curve, rec_curve, _ = precision_recall_curve(y_true_scores, scores_df[clf])
    # Выбираем точки, где recall >= 0.7
    valid_prec = prec_curve[rec_curve >= 0.7]
    if len(valid_prec) > 0:
        max_prec = valid_prec.max()
    else:
        max_prec = -1
    if max_prec > best_precision:
        best_precision = max_prec
        best_clf_for_prec = clf

print("\nЛучший по Precision при Recall >= 0.7:")
print(best_clf_for_prec)

print("\nМаксимальная точность (Precision) при Recall >= 0.7:")
print(f"{best_precision:.2f}")