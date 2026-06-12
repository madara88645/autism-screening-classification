from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "Autism.csv"
MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "autism_rf_pipeline.joblib"
METRICS_OUTPUT_PATH = ROOT_DIR / "outputs" / "metrics.json"
LOCAL_MLFLOW_URI = (ROOT_DIR / "mlruns").as_uri()

TARGET_COLUMN = "Class/ASD"
DROP_COLUMNS = ("age_desc",)
EXPERIMENT_NAME = "autism-screening-baseline"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
