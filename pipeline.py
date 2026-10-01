import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

os.makedirs('plots', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

df = pd.read_csv('winequality-red.csv')

df['is_good_quality'] = (df['quality'] >= 7).astype(int)

plt.figure(figsize=(7, 4))
ax = sns.countplot(
    data=df,
    x='is_good_quality',
    palette=['#3498db', '#2ecc71']
)
plt.title('Wine Target Stratification: Standard vs High Quality', fontsize=13, pad=12)
plt.xlabel('Quality Category')
plt.ylabel('Bottle Count')
ax.set_xticklabels(['Standard Quality (< 7)', 'High Quality (>= 7)'])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + 0.34, p.get_height() + 15), fontweight='bold')
plt.tight_layout()
plt.savefig('plots/01_quality_distribution.png')
plt.close()

plt.figure(figsize=(10, 8))
corr = df.drop(columns=['quality', 'is_good_quality']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5)
plt.title('Physicochemical Correlation Heatmap', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('plots/02_correlation_matrix.png')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(
    data=df,
    x='is_good_quality',
    y='alcohol',
    ax=axes[0],
    palette=['#e74c3c', '#2ecc71']
)
axes[0].set_title('Alcohol Content by Quality Tier', fontsize=12)
axes[0].set_xticklabels(['Standard', 'High'])
axes[0].set_xlabel('Quality Grade')
axes[0].set_ylabel('Alcohol (% vol)')

sns.boxplot(
    data=df,
    x='is_good_quality',
    y='volatile acidity',
    ax=axes[1],
    palette=['#e74c3c', '#2ecc71']
)
axes[1].set_title('Volatile Acidity by Quality Tier', fontsize=12)
axes[1].set_xticklabels(['Standard', 'High'])
axes[1].set_xlabel('Quality Grade')
axes[1].set_ylabel('Volatile Acidity (g/dm³)')
plt.tight_layout()
plt.savefig('plots/03_key_features_comparison.png')
plt.close()

df['total_acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
df['free_sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-5)
df['alcohol_to_sugar_ratio'] = df['alcohol'] / (df['residual sugar'] + 1e-5)

feature_columns = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol',
    'total_acidity', 'free_sulfur_ratio', 'alcohol_to_sugar_ratio'
]

X = df[feature_columns]
y = df['is_good_quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f'Model Training Successful!')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'ROC-AUC Score: {auc_score:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
    xticklabels=['Standard', 'High'],
    yticklabels=['Standard', 'High']
)
axes[0].set_title('Test Set Confusion Matrix', fontsize=12)
axes[0].set_xlabel('Predicted Quality')
axes[0].set_ylabel('True Quality')

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'ROC Curve (AUC = {auc_score:.3f})')
axes[1].plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--')
axes[1].set_title('Receiver Operating Characteristic', fontsize=12)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('plots/04_model_evaluation.png')
plt.close()

bundle = {
    'model': model,
    'features': feature_columns,
    'accuracy': accuracy,
    'auc': auc_score
}

joblib.dump(bundle, 'artifacts/wine_quality_model.joblib')
print('Artifact saved to artifacts/wine_quality_model.joblib')