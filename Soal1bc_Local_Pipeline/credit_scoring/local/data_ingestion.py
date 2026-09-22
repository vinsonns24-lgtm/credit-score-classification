from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
RAW_FILE = BASE_DIR / "data_C.csv"
INGESTED_DIR = BASE_DIR / "ingested"


class DataIngestion:
    """
    Step 1: Load raw CSV, validate, dan simpan ke folder ingested/.
    Pola sama dengan reference Model_Deployment_in_Streamlit.
    """

    def __init__(self, raw_file: Path = RAW_FILE, ingested_dir: Path = INGESTED_DIR):
        self.raw_file = Path(raw_file)
        self.ingested_dir = Path(ingested_dir)
        self.output_file = self.ingested_dir / "credit_score.csv"

    def run(self) -> Path:
        print("--- Step 1: Data Ingestion ---")
        self.ingested_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.raw_file, index_col=0)

        assert not df.empty, "❌ Dataset kosong!"
        assert "Credit_Score" in df.columns, "❌ Kolom target 'Credit_Score' tidak ditemukan!"

        print(f"  ✅ Loaded {len(df):,} baris × {df.shape[1]} kolom")
        print(f"  Distribusi target:\n{df['Credit_Score'].value_counts().to_string()}")

        df.to_csv(self.output_file, index=False)
        print(f"  Tersimpan → {self.output_file}\n")
        return self.output_file


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
