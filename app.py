import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Recreate functions used by the saved pipeline (from the training notebook)
# These must be present at import-time so pickle can find them when unpickling
def feature_engineering(X_df):
    X_df = X_df.copy()
    X_df['Total_Income'] = X_df['ApplicantIncome'] + X_df['CoapplicantIncome']
    return X_df.drop(['ApplicantIncome','CoapplicantIncome'], axis=1)

def log_transform(x):
    return np.log1p(x)

# Load the model (Ensure you saved the WHOLE pipeline, not just the classifier)
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_loan_approval(Gender, Married, Dependents, Education,
       Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
       Loan_Amount_Term, Credit_History, Property_Area):
    
    # 1. Create a DataFrame with the EXACT same column names used in X_train
    input_df = pd.DataFrame([[
        Gender, Married, Dependents, Education,
        Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Property_Area
    ]], columns=['Gender', 'Married', 'Dependents', 'Education',
                 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    # 2. Predict using the pipeline (the pipeline will handle feature engineering & scaling)
    prediction = model.predict(input_df)
    
    return "Approved ✅" if prediction[0] == 1 else "Not Approved ❌"

# Define input components (Removed Loan_Status)
inputs = [
    gr.Dropdown(choices=['Male', 'Female'], label="Gender"),
    gr.Dropdown(choices=['Yes', 'No'], label="Married"),
    gr.Dropdown(choices=['0', '1', '2', '3+'], label="Dependents"),
    gr.Dropdown(choices=['Graduate', 'Not Graduate'], label="Education"),
    gr.Dropdown(choices=['Yes', 'No'], label="Self Employed"),
    gr.Number(label="Applicant Income"),
    gr.Number(label="Coapplicant Income"),
    gr.Number(label="Loan Amount"),
    gr.Number(label="Loan Amount Term (e.g., 360)"),
    gr.Dropdown(choices=[1.0, 0.0], label="Credit History (1 for Good, 0 for Bad)"),
    gr.Dropdown(choices=['Urban', 'Semiurban', 'Rural'], label="Property Area"),
]

# Define Gradio interface
app = gr.Interface(
    fn=predict_loan_approval,
    inputs=inputs,
    outputs=gr.Textbox(label="Loan Prediction Result"),
    title="Loan Approval Prediction System",
    description="Enter applicant details to check loan eligibility."
)

app.launch(share=True)