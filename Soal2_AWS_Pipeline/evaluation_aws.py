"""
evaluation_aws.py — Evaluation script untuk SageMaker ProcessingStep.

Output: evaluation.json yang dibaca oleh SageMaker JsonGet (ConditionStep).
Format harus persis seperti ini agar SageMaker bisa membacanya.
"""

import os
import sys
import json
import tarfile
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    # Deteksi environment
    if os.path.exists("/opt/ml/processing"):
        model_dir  = "/opt/ml/processing/model"
        test_path  = "/opt/ml/processing/test/test.csv"
        output_dir = "/opt/ml/processing/evaluation"
    else:
        base       = "/home/ec2-user/SageMaker/credit_scoring"
        model_dir  = f"{base}/model"
        test_path  = f"{base}/test/test.csv"
        output_dir = f"{base}/eval"

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.pkl")

    # TrainingStep menyimpan artifact sebagai model.tar.gz (SageMaker selalu
    # mengompres isi SM_MODEL_DIR), bukan .pkl mentah. Kalau best_model.pkl
    # belum ada tapi model.tar.gz ada, extract dulu sebelum load.
    if not os.path.exists(model_path):
        tar_path = os.path.join(model_dir, "model.tar.gz")
        if os.path.exists(tar_path):
            print(f"📦 Menemukan {tar_path}, mengekstrak...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=model_dir)
            print(f"✅ Ekstraksi selesai.")

    if os.path.exists(model_path) and os.path.exists(test_path):
        model   = joblib.load(model_path)
        test_df = pd.read_csv(test_path)

        X_test  = test_df.drop("target", axis=1)
        y_test  = test_df["target"]

        y_pred  = model.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Format laporan yang dikenali SageMaker JsonGet
        report = {
            "multiclass_classification_metrics": {
                "accuracy": {"value": round(acc, 4), "standard_deviation": "NaN"},
                "f1_macro": {"value": round(f1,  4), "standard_deviation": "NaN"},
            }
        }

        out_file = os.path.join(output_dir, "evaluation.json")
        with open(out_file, "w") as f:
            json.dump(report, f)

        print(f"✅ Evaluation complete | Accuracy={acc:.4f} | F1 macro={f1:.4f}")
        print(f"   Report disimpan ke: {out_file}")
    else:
        if not os.path.exists(model_path):
            print(f"❌ Model tidak ada: {model_path}")
            print(f"   Isi {model_dir}: {os.listdir(model_dir) if os.path.exists(model_dir) else 'folder tidak ada'}")
        if not os.path.exists(test_path):
            print(f"❌ Test data tidak ada: {test_path}")
        # PENTING: exit dengan kode error, jangan biarkan exit 0 seolah sukses.
        # Kalau tidak, SageMaker (termasuk local mode) akan menandai step ini
        # "SUCCEEDED" padahal evaluation.json tidak pernah dibuat, dan step
        # berikutnya (ConditionStep / pengecekan threshold) akan gagal membaca
        # file yang sebenarnya tidak ada.
        sys.exit(1)