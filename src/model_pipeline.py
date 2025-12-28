from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_train import train_model
from src.model_evaluate import evaluate_model


def run_pipeline():
    print("Starting data ingestion...")
    df = data_ingestion()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = data_preprocessing(df)

    print("Training Gradient Boosting model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\nModel Training Completed")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    run_pipeline()
