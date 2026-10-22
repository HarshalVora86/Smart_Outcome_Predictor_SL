from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset/Smart_Outcome_Predictor_Dataset_5200.xlsx"

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

NUM = ["age", "week_of_year", "sessions", "time_spent_hours", "videos_watched", "quiz_attempts",
       "assignments_submitted", "forum_posts", "avg_quiz_score", "attendance_rate"]
ORD = ["course_level"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]
CAT = ["country_region", "device_type",
       "education_background", "course_category"]
# right-skewed columns capped with IQR bounds
CAP = ["sessions", "time_spent_hours", "videos_watched"]
FEATURES = NUM + ORD + CAT
TARGET_CLF = "completion_status"
TARGET_REG = "final_score"

df = pd.read_excel(DATA_PATH)
# ID and redundant with week_of_year
df = df.drop(columns=["student_id", "course_start_date"])

X = df[FEATURES].copy()
y_clf = df[TARGET_CLF]
y_reg = df[TARGET_REG]

X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clf, y_reg, test_size=0.20, random_state=42, stratify=y_clf
)

# Outlier capping: IQR bounds are learned from the training data only
cap_bounds = {}
for col in CAP:
    filled = X_train[col].fillna(X_train[col].median())
    q1, q3 = filled.quantile(0.25), filled.quantile(0.75)
    iqr = q3 - q1
    lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    cap_bounds[col] = (lower, upper)
    X_train[col] = X_train[col].clip(lower, upper)
    X_test[col] = X_test[col].clip(lower, upper)

# Preprocessing: median imputation, ordinal course level, one-hot for the other categories
pre = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), NUM),
    ("level", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[LEVELS])),
    ]), ORD),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first",
         handle_unknown="ignore", sparse_output=False)),
    ]), CAT),
])

# Base models: scaling happens inside the pipeline of each model that needs it
base_clfs = [
    ("lr", Pipeline([("scaler", StandardScaler()),
     ("lr", LogisticRegression(max_iter=1000))])),
    ("knn", Pipeline([("scaler", StandardScaler()),
     ("knn", KNeighborsClassifier(n_neighbors=15))])),
    ("svm", Pipeline([("scaler", StandardScaler()),
     ("svm", SVC(probability=True, random_state=42))])),
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
]
base_regs = [
    ("ridge", Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])),
    ("knn", Pipeline([("scaler", StandardScaler()),
     ("knn", KNeighborsRegressor(n_neighbors=15))])),
    ("svr", Pipeline([("scaler", StandardScaler()), ("svr", SVR())])),
    ("dt", DecisionTreeRegressor(max_depth=5, random_state=42)),
]

clf_pipeline = Pipeline([
    ("preprocessing", clone(pre)),
    ("ensemble", StackingClassifier(estimators=base_clfs,
                                    final_estimator=LogisticRegression(max_iter=1000), cv=5)),
])
reg_pipeline = Pipeline([
    ("preprocessing", clone(pre)),
    ("ensemble", StackingRegressor(estimators=base_regs, final_estimator=Ridge(), cv=5)),
])

print("Training Stacking Classifier ...")
clf_pipeline.fit(X_train, y_clf_train)
print("Training Stacking Regressor ...")
reg_pipeline.fit(X_train, y_reg_train)

yp = clf_pipeline.predict(X_test)
pr = clf_pipeline.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_clf_test, yp)
rp = reg_pipeline.predict(X_test)

metrics = {
    "classification": {
        "accuracy": float(accuracy_score(y_clf_test, yp)),
        "precision": float(precision_score(y_clf_test, yp, zero_division=0)),
        "recall": float(recall_score(y_clf_test, yp, zero_division=0)),
        "f1_score": float(f1_score(y_clf_test, yp, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_clf_test, pr)),
        "confusion_matrix": cm.tolist(),
        "ensemble": "Stacking Classifier",
        "base_models": ["Logistic Regression", "KNN", "SVM", "Decision Tree"],
        "meta_learner": "Logistic Regression",
    },
    "regression": {
        "mae": float(mean_absolute_error(y_reg_test, rp)),
        "rmse": float(np.sqrt(mean_squared_error(y_reg_test, rp))),
        "r2": float(r2_score(y_reg_test, rp)),
        "ensemble": "Stacking Regressor",
        "base_models": ["Ridge", "KNN", "SVR", "Decision Tree"],
        "meta_learner": "Ridge",
    },
    "train_rows": len(X_train),
    "test_rows": len(X_test),
}

bundle = {
    "classifier": clf_pipeline,
    "regressor": reg_pipeline,
    "cap_bounds": cap_bounds,
    "feature_columns": FEATURES,
}

with open(MODEL_DIR / "smart_outcome_predictor_pipeline.pkl", "wb") as f:
    pickle.dump(bundle, f)

(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4))

print(json.dumps(metrics, indent=4))
