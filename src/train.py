import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuration (You can adjust these as needed)
DATA_DIR = 'data'
MODEL_DIR = 'model'
TRAIN_SCRIPT_PATH = 'src/train.py'
APP_SCRIPT_PATH = 'src/app.py'
NOTEBOOK_PATH = 'notebooks/customer_churn_prediction.ipynb'

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(data_dir):
    data = pd.read_csv(os.path.join(data_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

    # Basic Cleaning
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    # Feature Engineering (Example)
    # data['TenureInMonths'] = data['tenure'] * 12  # Example: If you have tenure in years

    # Separate features (X) and target (y)
    X = data.drop(['customerID', 'Churn'], axis=1)
    y = data['Churn']

    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform the validation and test data
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor

# --- Model Training ---
def train_model(X_train, y_train, X_val, y_val):
    # Logistic Regression
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_val_pred_log_reg = log_reg.predict(X_val)

    print("Logistic Regression Validation Accuracy:", accuracy_score(y_val, y_val_pred_log_reg))
    print(classification_report(y_val, y_val_pred_log_reg))

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_val_pred_rf = rf_model.predict(X_val)

    print("Random Forest Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))
    print(classification_report(y_val, y_val_pred_rf))

    # XGBoost
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_val_pred_xgb = xgb_model.predict(X_val)

    print("XGBoost Validation Accuracy:", accuracy_score(y_val, y_val_pred_xgb))
    print(classification_report(y_val, y_val_pred_xgb))

    return log_reg, rf_model, xgb_model

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)

    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# --- Main Function ---
def main():
    X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor = load_and_preprocess_data(DATA_DIR)
    log_reg, rf_model, xgb_model = train_model(X_train_processed, y_train, X_val_processed, y_val)

    # Choose the best model based on validation performance
    best_model = xgb_model  # Replace with your chosen model

    evaluate_model(best_model, X_test_processed, y_test)

    # Save the model and preprocessor
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'churn_prediction_model.pkl'))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))

if __name__ == "__main__":
    main()
