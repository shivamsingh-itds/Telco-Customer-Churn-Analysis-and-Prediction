import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_preprocessing(df):

    # Make a safe copy
    df = df.copy()

    # Drop ID column
    df = df.drop(columns=["customerID"])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Binary mapping
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Internet service related columns
    internet_service_cols = [
        "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    for col in internet_service_cols:
        df[col] = df[col].replace({
            "Yes": 1,
            "No": 0,
            "No internet service": 0
        })

    # Phone service related column (THIS WAS THE BUG)
    df["MultipleLines"] = df["MultipleLines"].replace({
        "Yes": 1,
        "No": 0,
        "No phone service": 0
    })

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(
        df,
        columns=["Contract", "InternetService", "PaymentMethod", "gender"],
        drop_first=True
    )

    # Feature engineering
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Split data
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling numerical columns
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test
