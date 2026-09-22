import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)

TRACKING_URI = f"sqlite:///{(Path(__file__).parent / 'mlflow.db').as_posix()}"


class ModelEvaluator:
    """
    Step 4: Load model terbaik dari MLflow berdasarkan run_id,
    evaluasi pada test set, dan log semua metrik kembali ke MLflow.

    Menggunakan metrik macro-averaged karena dataset imbalanced (3 kelas).
    - Accuracy     : gambaran umum, tapi bisa misleading untuk imbalanced data
    - Precision    : seberapa tepat prediksi per kelas
    - Recall       : seberapa lengkap prediksi per kelas
    - F1 macro     : harmonic mean precision+recall, rata-rata tiap kelas (METRIK UTAMA)

    OOP class requirement untuk soal 1b.
    """

    LABEL_MAP = {0: "Good", 1: "Standard", 2: "Poor"}

    def run(
        self,
        run_id: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[float, float, float, float]:
        """
        Evaluasi model dari MLflow run_id pada test data.
        Return: (accuracy, precision_macro, recall_macro, f1_macro)
        """
        print("--- Step 4: Evaluation ---")

        mlflow.set_tracking_uri(TRACKING_URI)

        # Load model langsung dari MLflow tracking store
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        # Prediksi
        y_pred = model.predict(X_test)

        # Hitung semua metrik
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Log metrik kembali ke run MLflow yang sama
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_accuracy",         round(acc,  4))
            mlflow.log_metric("test_precision_macro",  round(prec, 4))
            mlflow.log_metric("test_recall_macro",     round(rec,  4))
            mlflow.log_metric("test_f1_macro",         round(f1,   4))

        # Print classification report per kelas
        target_names = [self.LABEL_MAP[i] for i in sorted(self.LABEL_MAP)]
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        print(f"  Test Accuracy       : {acc:.4f}")
        print(f"  Test Precision      : {prec:.4f}  (macro)")
        print(f"  Test Recall         : {rec:.4f}  (macro)")
        print(f"  Test F1 macro       : {f1:.4f}  (macro ← metrik utama)\n")

        return acc, prec, rec, f1


if __name__ == "__main__":
    print("Jalankan pipeline.py untuk eksekusi evaluasi lengkap.")
