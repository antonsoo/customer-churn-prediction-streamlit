# Paste the entire content of your Streamlit app (app.py) here
import streamlit as st
import pandas as pd
import joblib
import os

# Set the absolute path to the directory where models are stored
model_dir = "/content/drive/MyDrive/customer-churn-prediction/model"  # Update this to your actual path

# Load the trained model and preprocessor
try:
    model = joblib.load(os.path.join(model_dir, 'churn_prediction_model.pkl'))
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Function to preprocess input data
def preprocess_input(data, preprocessor):
    df = pd.DataFrame([data])
    df_processed = preprocessor.transform(df)
    return df_processed

# Streamlit app
st.title("Customer Churn Prediction App")
st.write("""
Enter customer information to predict churn.
""")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 1)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", 0.00, 200.00, 50.00)
total_charges = st.number_input("Total Charges", 0.00, 10000.00, 1000.00)
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Convert input to a dictionary
input_data = {
    "gender": gender,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "InternetService": internet_service
}

# Prediction button
if st.button("Predict Churn"):
    try:
        # Preprocess input data
        input_data_processed = preprocess_input(input_data, preprocessor)

        # Make prediction
        prediction = model.predict(input_data_processed)
        probability = model.predict_proba(input_data_processed)

        # Display result
        if prediction[0] == 1:
            st.warning("This customer is likely to churn.")
            st.write(f"Churn Probability: {probability[0][1]:.2f}")
        else:
            st.success("This customer is likely to stay.")
            st.write(f"Churn Probability: {probability[0][0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
