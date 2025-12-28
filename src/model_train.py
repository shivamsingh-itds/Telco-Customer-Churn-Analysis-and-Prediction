from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingClassifier


def train_model(X_train, y_train):

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "gradient_boosting_model.pkl"
    joblib.dump(model, model_path)

    return model
