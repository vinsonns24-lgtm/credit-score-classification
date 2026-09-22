import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

from preprocessing import CreditScorePreprocessor

BASE_DIR      = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TRACKING_URI  = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


class CreditModelTrainer:
    """
    Step 3: Hyperparameter tuning (GridSearchCV) untuk beberapa model ML,
    log SETIAP kombinasi hyperparameter sebagai nested run ke MLflow,
    lalu simpan model terbaik.

    OOP class requirement untuk soal 1b.
    """

    # (nama_model, classifier, param_grid)
    # Grid sengaja dibuat cukup kecil supaya total waktu training tetap wajar
    # (tiap kombinasi = 1x training 3-fold CV). Silakan diperbesar sesuai
    # kebutuhan / waktu yang tersedia.
    MODELS: List[Tuple[str, Any, Dict]] = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {
                "classifier__max_depth": [5, 10, 15, 20],
                "classifier__min_samples_split": [2, 5, 10],
            },
        ),
        (
            "RandomForest",
            # n_jobs tidak di-set -1 supaya RAM tetap aman (GridSearchCV juga n_jobs=1)
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [8, 12, 16],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            {
                "classifier__max_depth": [3, 5],
            },
        ),
    ]

    # Nama eksperimen baru, supaya run lama (split acak per baris, skornya
    # sedikit terlalu tinggi) tidak tercampur dengan run versi split per nasabah.
    def __init__(self, experiment_name: str = "Credit Score Classification - split per nasabah"):
        self.experiment_name = experiment_name
        self.preprocessor = CreditScorePreprocessor()
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)

    def train_all(self, data_path: Path) -> Tuple[str, pd.DataFrame, pd.Series]:
        """
        Tuning + training semua model, log ke MLflow (parent + nested runs),
        simpan best_model.pkl.
        Return: (best_run_id, X_test, y_test)
        """
        print("--- Step 3: Hyperparameter Tuning + Training ---")

        # Siapkan data (sudah dibagi per nasabah)
        X_train, X_test, y_train, y_test, groups_train = self.preprocessor.clean_and_split(data_path)
        column_transformer = self.preprocessor.build_column_transformer()

        best_run_id = None
        best_cv_f1 = -1.0
        # CV juga per nasabah: satu nasabah tidak boleh ada di fold latih dan fold validasi sekaligus
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        for model_name, classifier, param_grid in self.MODELS:
            n_combos = 1
            for values in param_grid.values():
                n_combos *= len(values)
            print(f"\n  ▶ Tuning [{model_name}] — {n_combos} kombinasi hyperparameter × {cv.get_n_splits()}-fold CV")

            # Build Unified sklearn Pipeline: ColumnTransformer → Classifier
            # clone() penting agar setiap model dapat transformer yang fresh (belum di-fit)
            full_pipeline = SklearnPipeline([
                ("preprocessor", clone(column_transformer)),
                ("classifier",   classifier),
            ])

            with mlflow.start_run(run_name=model_name) as parent_run:
                mlflow.log_param("model", model_name)
                mlflow.log_param("n_hyperparam_combinations", n_combos)
                mlflow.log_param("cv_folds", cv.get_n_splits())
                mlflow.log_param("split", "per nasabah (Customer_ID)")

                # GridSearchCV: coba semua kombinasi hyperparameter,
                # tiap kombinasi dievaluasi dengan 3-fold Stratified Group CV
                # pada TRAINING SET saja (X_test/y_test tidak disentuh).
                search = GridSearchCV(
                    estimator=full_pipeline,
                    param_grid=param_grid,
                    scoring="f1_macro",
                    cv=cv,
                    n_jobs=1,
                    refit=True,          # otomatis fit ulang best_params_ ke seluruh X_train
                    return_train_score=False,
                )
                search.fit(X_train, y_train, groups=groups_train)

                # ─── Log SETIAP kombinasi hyperparameter sebagai nested run ──────
                cv_results = search.cv_results_
                for i in range(len(cv_results["params"])):
                    with mlflow.start_run(run_name=f"{model_name}_trial_{i}", nested=True):
                        for raw_name, value in cv_results["params"][i].items():
                            clean_name = raw_name.replace("classifier__", "")
                            mlflow.log_param(clean_name, value)
                        mlflow.log_metric("cv_f1_macro_mean", round(float(cv_results["mean_test_score"][i]), 4))
                        mlflow.log_metric("cv_f1_macro_std",  round(float(cv_results["std_test_score"][i]), 4))
                        mlflow.log_metric("rank_cv_f1_macro", int(cv_results["rank_test_score"][i]))

                # ─── Kombinasi terbaik untuk model ini ────────────────────────────
                best_params = {
                    k.replace("classifier__", ""): v for k, v in search.best_params_.items()
                }
                best_cv_f1_this_model = float(search.best_score_)

                for k, v in best_params.items():
                    mlflow.log_param(f"best_{k}", v)
                mlflow.log_metric("best_cv_f1_macro", round(best_cv_f1_this_model, 4))

                # search.best_estimator_ sudah di-refit otomatis (refit=True) ke seluruh X_train
                best_pipeline_this_model = search.best_estimator_

                # Train F1 tetap dicatat sebagai referensi (bukan kriteria seleksi)
                y_train_pred = best_pipeline_this_model.predict(X_train)
                train_f1 = f1_score(y_train, y_train_pred, average="macro")
                mlflow.log_metric("train_f1_macro", round(train_f1, 4))

                # Log model terbaik (hasil tuning) untuk algoritma ini ke MLflow
                mlflow.sklearn.log_model(
                    best_pipeline_this_model,
                    artifact_path="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                )
                run_id = parent_run.info.run_id

                print(f"    Run ID            : {run_id}")
                print(f"    Best params       : {best_params}")
                print(f"    Best CV F1 macro  : {best_cv_f1_this_model:.4f}")
                print(f"    Train F1 macro    : {train_f1:.4f}  (referensi, bukan kriteria seleksi)")

                # Simpan jika ini model terbaik antar-algoritma sejauh ini
                if best_cv_f1_this_model > best_cv_f1:
                    best_cv_f1 = best_cv_f1_this_model
                    best_run_id = run_id
                    joblib.dump(best_pipeline_this_model, ARTIFACTS_DIR / "best_model.pkl")
                    print(f"    ⭐ Best model baru! Disimpan ke artifacts/best_model.pkl")

        print(f"\n✅ Tuning + Training selesai. Best run: {best_run_id} (CV F1 macro={best_cv_f1:.4f})\n")
        return best_run_id, X_test, y_test


if __name__ == "__main__":
    trainer = CreditModelTrainer()
    run_id, X_test, y_test = trainer.train_all(BASE_DIR / "ingested" / "credit_score.csv")
    print(f"Run ID terbaik: {run_id}")