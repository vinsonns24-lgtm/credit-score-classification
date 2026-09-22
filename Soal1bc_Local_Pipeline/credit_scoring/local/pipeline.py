from pathlib import Path
from data_ingestion import DataIngestion
from train import CreditModelTrainer
from evaluation import ModelEvaluator

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data_C.csv"
F1_THRESHOLD = 0.65   # Minimum F1 macro untuk approve deployment


class CreditScoringPipeline:
    """
    Master pipeline yang mengorkestrasikan semua step Credit Score Classification.

    Alur:
    Step 1 → DataIngestion     : load & validasi raw CSV
    Step 2 → CreditModelTrainer: cleaning + split + training semua model (via preprocessing.py)
    Step 3 → ModelEvaluator    : evaluasi model terbaik pada test set
    Step 4 → Deployment Gate   : approve jika F1 ≥ threshold

    Pola diambil dari ML_OOP ChurnPredictionPipeline reference.
    """

    def __init__(self, data_path: Path = DATA_PATH, f1_threshold: float = F1_THRESHOLD):
        self.data_path    = data_path
        self.f1_threshold = f1_threshold

        # Instantiasi semua OOP class
        self.ingestor  = DataIngestion(raw_file=self.data_path)
        self.trainer   = CreditModelTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print("=" * 60)
        print("  🚀  Credit Scoring Pipeline — Starting")
        print("=" * 60)

        # Step 1: Ingest raw data
        ingested_file = self.ingestor.run()

        # Step 2+3: Preprocess + Train semua model
        #   (preprocessing dilakukan di dalam CreditModelTrainer.train_all()
        #    via CreditScorePreprocessor.clean_and_split())
        best_run_id, X_test, y_test = self.trainer.train_all(ingested_file)

        # Step 4: Evaluate model terbaik pada test set
        acc, prec, rec, f1 = self.evaluator.run(best_run_id, X_test, y_test)

        # Step 5: Deployment approval gate
        print("--- Deployment Approval ---")
        if f1 >= self.f1_threshold:
            print(f"✅ APPROVED — F1 macro ({f1:.4f}) ≥ threshold ({self.f1_threshold})")
            print("   Model tersimpan di : artifacts/best_model.pkl")
            print("   Untuk deploy lokal : streamlit run app.py")
        else:
            print(f"❌ REJECTED — F1 macro ({f1:.4f}) < threshold ({self.f1_threshold})")
            print("   Coba tuning hyperparameter atau tambah fitur.")

        print("=" * 60)


if __name__ == "__main__":
    pipeline = CreditScoringPipeline()
    pipeline.execute()
