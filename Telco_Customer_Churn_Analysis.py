import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('Telco Customer Churn.csv')
data = data.replace(r'^\s*$', np.nan, regex=True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.fillna(data['MonthlyCharges'] * np.maximum(data['tenure'], 1), inplace=True)
data.drop_duplicates(inplace=True)

# Encode categorical variables
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
categorical_vars = [col for col in data.columns if data[col].dtype == 'O']
for var in categorical_vars:
    data[var] = data[var].map(data.groupby(var)['Churn'].mean().rank().to_dict())

# Balance the dataset
X = data.drop(['Churn'], axis=1)
y = data['Churn']
X, y = BorderlineSMOTE().fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_features = ['TotalCharges', 'MonthlyCharges', 'tenure']
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Define models and stacking
models = [
    ('Logistic Regression', LogisticRegression(random_state=42, penalty='l2', class_weight='balanced', C=6)),
    ('Random Forest', RandomForestClassifier(n_estimators=70, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=1)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=90, learning_rate=1, random_state=42)),
    ('XGBoost', XGBClassifier(n_estimators=120, learning_rate=1, random_state=42)),
    ('Extra Trees', ExtraTreesClassifier(n_estimators=140, random_state=42))
]

stacking_clf = StackingClassifier(
    estimators=models,
    final_estimator=LogisticRegression(random_state=42)
)

# Evaluate and fit the stacking model
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f'{name} Accuracy: {scores.mean():.4f}')

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)
y_prob = stacking_clf.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()