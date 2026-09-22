"""
data_ingestion_aws.py — Script ingestion untuk SageMaker ProcessingStep.

Pola dari Training_Pipeline reference:
- Deteksi otomatis apakah berjalan di SageMaker container atau lokal
- Di container  : /opt/ml/processing/input  → /opt/ml/processing/ingested
- Di SageMaker NB lokal: /home/ec2-user/SageMaker/credit_scoring/...
"""

import os
import pandas as pd

def ingest_data():
    # Deteksi environment: SageMaker container vs lokal
    if os.path.exists("/opt/ml/processing/input"):
        input_dir  = "/opt/ml/processing/input"
        output_dir = "/opt/ml/processing/ingested"
    else:
        input_dir  = "/home/ec2-user/SageMaker/credit_scoring"
        output_dir = "/home/ec2-user/SageMaker/credit_scoring/ingested"

    os.makedirs(output_dir, exist_ok=True)

    input_file  = os.path.join(input_dir, "data_C.csv")
    output_file = os.path.join(output_dir, "credit_score.csv")

    print(f"Looking for file at: {input_file}")

    if os.path.exists(input_file):
        df = pd.read_csv(input_file, index_col=0)

        assert not df.empty, "Dataset kosong!"
        assert "Credit_Score" in df.columns, "Kolom target tidak ditemukan!"

        print(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
        print(f"Target distribution:\n{df['Credit_Score'].value_counts()}")

        df.to_csv(output_file, index=False)
        print(f"✅ Disimpan ke: {output_file}")
    else:
        print(f"❌ File tidak ditemukan: {input_file}")
        print(f"Files di {input_dir}: {os.listdir(input_dir) if os.path.exists(input_dir) else 'folder tidak ada'}")


if __name__ == "__main__":
    ingest_data()
