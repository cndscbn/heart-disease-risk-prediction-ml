# ==========================================================
# PUBLICATION-GRADE HEART DISEASE PREDICTION PIPELINE
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy import stats

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# ==========================================================
# EXTERNAL VALIDATION SPLIT
# ==========================================================

X_train, X_ext, y_train, y_ext = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================================
# PIPELINE BUILDER
# ==========================================================

def build_pipeline(model):
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", model)
    ])

# ==========================================================
# BASELINE MODELS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Random Forest": RandomForestClassifier(n_estimators=400),
    "SVM (RBF)": SVC(kernel='rbf', probability=True)
}

# ==========================================================
# HYPERPARAMETER TUNING (XGBoost)
# ==========================================================

xgb = XGBClassifier(eval_metric='logloss')

param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid = GridSearchCV(xgb, param_grid, cv=5,
                    scoring='roc_auc', n_jobs=-1)

pipe_xgb = build_pipeline(grid)
pipe_xgb.fit(X_train, y_train)
best_xgb = pipe_xgb.named_steps["classifier"].best_estimator_

# ==========================================================
# STRATIFIED 10-FOLD CV
# ==========================================================

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    pipe = build_pipeline(model)
    auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')
    acc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='accuracy')
    results.append({
        "Model": name,
        "Median Accuracy": np.median(acc),
        "Median AUC": np.median(auc)
    })

# Tuned XGBoost
auc_xgb = cross_val_score(build_pipeline(best_xgb),
                          X_train, y_train, cv=skf, scoring='roc_auc')

acc_xgb = cross_val_score(build_pipeline(best_xgb),
                          X_train, y_train, cv=skf, scoring='accuracy')

results.append({
    "Model": "Tuned XGBoost",
    "Median Accuracy": np.median(acc_xgb),
    "Median AUC": np.median(auc_xgb)
})

results_df = pd.DataFrame(results)

baseline_auc = results_df.loc[
    results_df["Model"]=="Logistic Regression","Median AUC"
].values[0]

results_df["AUC Ratio vs LR"] = results_df["Median AUC"] / baseline_auc
results_df.to_csv("Table_A_Results.csv", index=False)

# ==========================================================
# EXTERNAL VALIDATION
# ==========================================================

final_pipe = build_pipeline(best_xgb)
final_pipe.fit(X_train, y_train)

y_prob = final_pipe.predict_proba(X_ext)[:,1]
y_pred = final_pipe.predict(X_ext)

ext_auc = roc_auc_score(y_ext, y_prob)
ext_acc = accuracy_score(y_ext, y_pred)

# ==========================================================
# BOOTSTRAP 95% CI
# ==========================================================

boot_scores = []
for i in range(1000):
    idx = resample(range(len(y_ext)))
    if len(np.unique(y_ext.iloc[idx])) < 2:
        continue
    score = roc_auc_score(y_ext.iloc[idx], y_prob[idx])
    boot_scores.append(score)

ci_lower = np.percentile(boot_scores, 2.5)
ci_upper = np.percentile(boot_scores, 97.5)

# ==========================================================
# DELONG TEST
# ==========================================================

def delong_test(y_true, pred1, pred2):
    diff = roc_auc_score(y_true, pred1) - roc_auc_score(y_true, pred2)
    se = np.std(boot_scores)
    z = diff / se
    p = 2*(1 - stats.norm.cdf(abs(z)))
    return p

log_pipe = build_pipeline(LogisticRegression(max_iter=3000))
log_pipe.fit(X_train, y_train)
log_probs = log_pipe.predict_proba(X_ext)[:,1]

p_value = delong_test(y_ext, y_prob, log_probs)

# ==========================================================
# CALIBRATION CURVE
# ==========================================================

prob_true, prob_pred = calibration_curve(y_ext, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--')
plt.title("Calibration Curve")
plt.savefig("Calibration.png")
plt.close()

# ==========================================================
# DECISION CURVE
# ==========================================================

thresholds = np.linspace(0.01,0.99,100)
net_benefit = []

for t in thresholds:
    preds = (y_prob >= t).astype(int)
    tp = np.sum((preds==1) & (y_ext==1))
    fp = np.sum((preds==1) & (y_ext==0))
    n = len(y_ext)
    nb = (tp/n) - (fp/n)*(t/(1-t))
    net_benefit.append(nb)

plt.plot(thresholds, net_benefit)
plt.title("Decision Curve")
plt.savefig("Decision_Curve.png")
plt.close()

# ==========================================================
# FAIRNESS CHECK
# ==========================================================

if "sex" in X.columns:
    male = X_ext["sex"]==1
    female = X_ext["sex"]==0
    male_auc = roc_auc_score(y_ext[male], y_prob[male])
    female_auc = roc_auc_score(y_ext[female], y_prob[female])

# ==========================================================
# SHAP
# ==========================================================

explainer = shap.Explainer(final_pipe.named_steps["classifier"])
shap_values = explainer(X_ext.sample(100))
shap.plots.beeswarm(shap_values)

print("External AUC: - heart_disease_full_publication_pipeline.py:220", ext_auc)
print("95% CI: - heart_disease_full_publication_pipeline.py:221", ci_lower, "-", ci_upper)
print("DeLong pvalue: - heart_disease_full_publication_pipeline.py:222", p_value)
print("Project Complete - heart_disease_full_publication_pipeline.py:223")