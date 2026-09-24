"""
train_aws.py — Training script untuk SageMaker TrainingStep.

Sama dengan train.py versi lokal:
- 4 model, masing-masing di-tuning dengan GridSearchCV (grid yang sama)
- Cross-validation 3-fold per nasabah (StratifiedGroupKFold)
- Metrik pemilihan: F1 macro
- Preprocessing (imputasi, scaling, one-hot) digabung dengan classifier dalam
  satu sklearn Pipeline, lalu disimpan sebagai best_model.pkl

Bedanya dengan versi lokal: tidak ada MLflow. Ringkasan tuning disimpan di
tuning_summary.json di samping model.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

NUMERIC_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age_Months",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
]

CATEGORICAL_FEATURES = [
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"
]

# 1. Preprocessor: median + scaling untuk numerik, modus + one-hot untuk kategorikal.
#    Occupation dan Payment_Behaviour tidak punya urutan, jadi one-hot lebih tepat
#    daripada ordinal encoding.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
], remainder="drop")

# 2. Model dan grid, sama dengan train.py lokal (prefix "classifier__" untuk Pipeline)
MODELS = [
    (
        "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    (
        "DecisionTree",
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"classifier__max_depth": [5, 10, 15, 20], "classifier__min_samples_split": [2, 5, 10]},
    ),
    (
        "RandomForest",
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"classifier__n_estimators": [50, 100, 200], "classifier__max_depth": [8, 12, 16]},
    ),
    (
        "GradientBoosting",
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        {"classifier__max_depth": [3, 5]},
    ),
]

def main():
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/home/ec2-user/SageMaker/credit_scoring/train")
    model_dir = os.environ.get("SM_MODEL_DIR",     "/home/ec2-user/SageMaker/credit_scoring/model")
    os.makedirs(model_dir, exist_ok=True)

    train_file = os.path.join(train_dir, "train.csv")
    if not os.path.exists(train_file):
        print(f"❌ Training file tidak ditemukan: {train_file}")
        return

    df = pd.read_csv(train_file)
    groups  = df["customer_id"]
    X_train = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = df["target"]

    # CV per nasabah: satu nasabah tidak boleh ada di fold latih dan fold validasi sekaligus
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    best_model = None
    best_model_name = None
    best_params = None
    best_cv_f1 = -1.0
    summary = []

    for model_name, classifier, param_grid in MODELS:
        n_combos = 1
        for values in param_grid.values():
            n_combos *= len(values)
        print(f"\n▶ Tuning [{model_name}] — {n_combos} kombinasi × {cv.get_n_splits()}-fold CV")

        # 3. Gabungkan preprocessor dan algoritma dalam satu Pipeline utuh
        full_pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier)
        ])

        # n_jobs=2 supaya RAM notebook instance tetap aman
        search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=2,
            refit=True,
        )
        search.fit(X_train, y_train, groups=groups)

        cv_f1 = float(search.best_score_)
        print(f"  Best params      : {search.best_params_}")
        print(f"  Best CV F1 macro : {cv_f1:.4f}")

        summary.append({
            "model": model_name,
            "best_params": search.best_params_,
            "cv_f1_macro": round(cv_f1, 4),
        })

        if cv_f1 > best_cv_f1:
            best_cv_f1 = cv_f1
            best_model = search.best_estimator_
            best_model_name = model_name
            best_params = search.best_params_

    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)

    summary_path = os.path.join(model_dir, "tuning_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "winner": best_model_name,
            "winner_params": best_params,
            "winner_cv_f1_macro": round(best_cv_f1, 4),
            "all_models": summary,
        }, f, indent=2)

    print(f"\n✅ Training complete.")
    print(f"   Model terbaik      : {best_model_name} (CV F1 macro={best_cv_f1:.4f})")
    print(f"   Best hyperparams   : {best_params}")
    print(f"   Model disimpan ke  : {model_path}")
    print(f"   Ringkasan tuning   : {summary_path}")

if __name__ == "__main__":
    main()
