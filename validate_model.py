import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score,precision_score,roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib 
import sys

print("Starting Model Validaiton")

df = pd.read_csv("creditcard_sample.csv")
X = df.drop('Class', axis=1)
y = df['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#could have avoided using smote with xgboost but using it because it was used in notebook before 
#use scale_pos_weight instead with xgboost
smote = SMOTE(random_state=42)
X_train_bal,y_train_bal = smote.fit_resample(X_train,y_train)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.2,
        random_state=42,
        min_child_weight=10
    ))
])

pipe.fit(X_train_bal,y_train_bal)

predictions = pipe.predict(X_test)
pred_proba = pipe.predict_proba(X_test)[:,1]

recall = recall_score(y_test,predictions)
precision = precision_score(y_test,predictions)
auc = roc_auc_score(y_test,pred_proba)

print(f"Recall is: {recall:.4f}")
print(f"Precision is: {precision:.4f}")
print(f"Roc_auc is: {auc:.4f}")

RECALL_THRESHOLD = 0.75
AUC_THRESHOLD = 0.96

if recall < RECALL_THRESHOLD:
    print(f"FAILED RECALL SCORE IS BELOW THRESHOLD!!!, SCORE WAS {recall:.4f} ")
    sys.exit(1)

if auc<AUC_THRESHOLD:
    print(f"FAILED ROC_AUC SCORE IS BELOW THRESHOLD!!!, AUC SCORE WAS {auc:.4f}")
    sys.exit(1)

with open('fraud_model.pkl', 'wb') as f:
    joblib.dump(pipe,f)

print("PASSED: ALL METRICS WERE ABOVE THRESHOLD")
print("MODEL SAVED AS fraud_model.pkl")

