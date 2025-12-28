from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def data_ingestion():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "Telco-Customer-Churn.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    return df
