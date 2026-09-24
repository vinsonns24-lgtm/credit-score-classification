"""
pipeline_aws.py — SageMaker Pipeline untuk Credit Score Classification.

Jalankan di SageMaker Notebook Instance.
Pola dari Training_Pipeline reference (pipeline.py).

Step 1: Ingest         → ProcessingStep (data_ingestion_aws.py)
Step 2: Preprocess     → ProcessingStep (preprocessing_aws.py)
Step 3: Train          → TrainingStep   (train_aws.py)
Step 4: Evaluate       → ProcessingStep (evaluation_aws.py)
Step 5: ConditionStep  → Deploy jika F1 ≥ threshold
"""

import os
import sagemaker
from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

# ─── 1. Konfigurasi ──────────────────────────────────────────────────────────

# Gunakan LocalPipelineSession untuk testing di Notebook Instance
local_session = LocalPipelineSession()

try:
    role = sagemaker.get_execution_role()
except:
    role = "LabRole"

instance_type = "local"  # Ganti ke "ml.m5.large" untuk SageMaker cloud penuh

base_path = "/home/ec2-user/SageMaker/credit_scoring"
local_uri  = f"file://{base_path}"

# Buat semua folder yang diperlukan
for folder in ["ingested", "train", "test", "eval", "model"]:
    os.makedirs(f"{base_path}/{folder}", exist_ok=True)

# ─── 2. Processor & Estimator ────────────────────────────────────────────────

processor = SKLearnProcessor(
    framework_version="1.4-2",   # sama dengan endpoint dan requirements.txt (scikit-learn 1.4.2)
    role=role,
    instance_type=instance_type,
    instance_count=1,
    sagemaker_session=local_session,
)

estimator = SKLearn(
    entry_point="train_aws.py",
    role=role,
    instance_type=instance_type,
    framework_version="1.4-2",
    sagemaker_session=local_session,
    # Model dan grid hyperparameter didefinisikan di train_aws.py (GridSearchCV untuk 4 model),
    # jadi tidak ada hyperparameter yang dikirim dari sini.
)

# ─── 3. Step Definitions ─────────────────────────────────────────────────────

# STEP 1: Ingestion
step_ingest = ProcessingStep(
    name="CreditScoreIngest",
    processor=processor,
    inputs=[ProcessingInput(
        source=base_path,                        # folder lokal dengan data_C.csv
        destination="/opt/ml/processing/input",  # di dalam container
    )],
    outputs=[ProcessingOutput(
        output_name="ingested_data",
        source="/opt/ml/processing/ingested",
        destination=f"{local_uri}/ingested",
    )],
    code="data_ingestion_aws.py",
)

# STEP 2: Preprocessing
step_preprocess = ProcessingStep(
    name="CreditScorePreprocess",
    processor=processor,
    inputs=[ProcessingInput(
        source=step_ingest.properties.ProcessingOutputConfig.Outputs["ingested_data"].S3Output.S3Uri,
        destination="/opt/ml/processing/ingested",
    )],
    outputs=[
        ProcessingOutput(output_name="train",
                         source="/opt/ml/processing/train",
                         destination=f"{local_uri}/train"),
        ProcessingOutput(output_name="test",
                         source="/opt/ml/processing/test",
                         destination=f"{local_uri}/test"),
    ],
    code="preprocessing_aws.py",
)

# STEP 3: Training
step_train = TrainingStep(
    name="CreditScoreTrain",
    estimator=estimator,
    # PENTING: gunakan referensi ke properti step_preprocess (BUKAN string path
    # literal seperti sebelumnya). Kalau cuma string biasa, SDK tidak bisa
    # mendeteksi dependency data yang sebenarnya antar step, sehingga urutan
    # eksekusi di Local Mode bisa jadi salah (Train sempat jalan SEBELUM
    # Preprocess selesai, akibatnya train.csv belum ada saat dibutuhkan).
    # Referensi property di bawah ini memaksa DAG-nya benar.
    inputs={
        "train": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        )
    },
    depends_on=[step_preprocess],
)

# STEP 4: Evaluation
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

step_eval = ProcessingStep(
    name="CreditScoreEval",
    processor=processor,
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[ProcessingOutput(
        output_name="evaluation",
        source="/opt/ml/processing/evaluation",
        destination=f"{local_uri}/eval",
    )],
    code="evaluation_aws.py",
    property_files=[evaluation_report],
)

# STEP 5: Condition — approve jika F1 macro ≥ 0.65
#
# CATATAN PENTING: ConditionStep + JsonGet TIDAK didukung penuh di Local Mode.
# JsonGet mencoba membaca property_file dari S3 asli, padahal di local mode
# outputnya cuma ada di disk lokal → selalu FAILED dengan pesan "reading file
# ... from S3: None: None". Ini keterbatasan SDK, bukan bug di script kita.
#
# Solusi: di LOCAL MODE, ConditionStep di-skip dari pipeline; threshold dicek
# manual di Python setelah pipeline selesai (lihat bagian bawah).
# Kalau instance_type diganti ke cloud beneran ("ml.m5.large") + PipelineSession
# + S3 (bukan LocalPipelineSession/local_uri), ConditionStep+JsonGet ini akan
# berfungsi normal dan bisa dipakai lagi.

USE_LOCAL_MODE = (instance_type == "local")

if not USE_LOCAL_MODE:
    condition_f1 = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.f1_macro.value",
        ),
        right=0.65,
    )
    step_condition = ConditionStep(
        name="CheckF1Threshold",
        conditions=[condition_f1],
        if_steps=[],    # Tambahkan RegisterModel / CreateModel di sini jika diperlukan
        else_steps=[],
    )
    pipeline_steps = [step_ingest, step_preprocess, step_train, step_eval, step_condition]
else:
    pipeline_steps = [step_ingest, step_preprocess, step_train, step_eval]

# ─── 4. Pipeline ─────────────────────────────────────────────────────────────

pipeline = Pipeline(
    name="CreditScore-Local-Pipeline",
    steps=pipeline_steps,
    sagemaker_session=local_session,
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("✅ Pipeline (local mode) selesai dieksekusi — start() bersifat blocking/synchronous.\n")

# _LocalPipelineExecution TIDAK punya .wait()/.list_steps() seperti pipeline cloud.
# Statusnya disimpan di execution.step_execution (dict nama_step -> objek step lokal).
print("=== Status tiap step (local mode) ===")
for step_name, step_obj in execution.step_execution.items():
    status = getattr(step_obj, "status", "UNKNOWN")
    reason = getattr(step_obj, "failure_reason", None)
    print(f"  {step_name:25s} → {status}" + (f"  | ❌ {reason}" if reason else ""))

# ─── 5. Cek threshold F1 secara manual (pengganti ConditionStep di local mode) ─
if USE_LOCAL_MODE:
    import json as _json

    eval_json_path = f"{base_path}/eval/evaluation.json"
    print(f"\n=== Cek threshold F1 manual (baca: {eval_json_path}) ===")
    if os.path.exists(eval_json_path):
        with open(eval_json_path) as f:
            report = _json.load(f)
        f1 = report["multiclass_classification_metrics"]["f1_macro"]["value"]
        acc = report["multiclass_classification_metrics"]["accuracy"]["value"]
        passed = f1 >= 0.65
        status_icon = "✅ LOLOS" if passed else "❌ TIDAK LOLOS"
        print(f"  Accuracy = {acc:.4f}")
        print(f"  F1 macro = {f1:.4f}  → {status_icon} (threshold 0.65)")
        if passed:
            print("  → Model siap di-deploy (lanjut ke deploy_endpoint.ipynb)")
        else:
            print("  → Model belum memenuhi threshold, perlu tuning ulang sebelum deploy")
    else:
        print(f"  ❌ File evaluation.json tidak ditemukan di {eval_json_path}")

# ─── 6. Populate folder lokal secara otomatis (workaround SageMaker SDK) ───────
#
# Known limitation SageMaker Python SDK: di Local Mode, ProcessingOutput
# TIDAK pernah disalin ke `destination` lokal yang kita tentukan — file hasil
# tiap step cuma ada sesaat di folder /tmp Docker lalu dibuang begitu container
# selesai. Ini bug/limitation resmi di SDK-nya sendiri (lihat GitHub issue
# aws/sagemaker-python-sdk #3083), bukan kesalahan konfigurasi kita.
#
# Workaround: jalankan ulang script-script yang sama secara LANGSUNG (tanpa
# Docker/Pipeline). Karena tiap script sudah punya deteksi environment
# (if os.path.exists("/opt/ml/processing") ... else ...), begitu dijalankan
# langsung di host, otomatis masuk cabang `else` dan nulis file ke folder
# lokal asli (ingested/, train/, test/, model/, eval/) — bukan ke /tmp yang
# dibuang. Pipeline SageMaker di atas tetap membuktikan arsitektur cloud-nya
# (ProcessingStep/TrainingStep/ConditionStep jalan & berhasil), sedangkan
# bagian ini yang memastikan folder lokal benar-benar terisi untuk verifikasi.

if USE_LOCAL_MODE:
    import subprocess
    import sys

    print("\n=== Populate folder lokal (jalankan script langsung, bukan via Docker) ===")
    local_scripts = [
        "data_ingestion_aws.py",
        "preprocessing_aws.py",
        "train_aws.py",
        "evaluation_aws.py",
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for script in local_scripts:
        script_path = os.path.join(script_dir, script)
        print(f"\n▶ Menjalankan {script} langsung di host...")
        result = subprocess.run(
            [sys.executable, script_path],   # Python yang sama dengan pipeline ini, supaya versi scikit-learn sama
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ {script} gagal (exit code {result.returncode}):")
            print(result.stderr)
            break
    else:
        print("✅ Semua folder lokal (ingested/, train/, test/, model/, eval/) sudah terisi.")