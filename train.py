import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gradio as gr

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def feature_engineering(X_df):
    """Creates Total_Income and drops individual income columns."""
    X_df = X_df.copy()
    # Ensure numeric types
    X_df['ApplicantIncome'] = pd.to_numeric(X_df['ApplicantIncome'], errors='coerce')
    X_df['CoapplicantIncome'] = pd.to_numeric(X_df['CoapplicantIncome'], errors='coerce')
    X_df['Total_Income'] = X_df['ApplicantIncome'] + X_df['CoapplicantIncome']
    return X_df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

def log_transform(x):
    """Applies log(1+x) transformation to normalize skewed data."""
    return np.log1p(x)

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================

# Load Data
df = pd.read_csv("loan.csv") # Ensure your file is named loan.csv

# Define Features and Target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
le = LabelEncoder()
y = le.fit_transform(df['Loan_Status']) # Y -> 1, N -> 0

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Column Lists
# Note: Total_Income is created by feature_engineering, so it's in num_cols
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            'Property_Area', 'Credit_History', 'Loan_Amount_Term']
num_cols = ['Total_Income', 'LoanAmount']

# Categorical Pipeline
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Numerical Pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transformer', FunctionTransformer(log_transform)),
    ('scaler', StandardScaler())
])

# Combine into Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
    ]
)

# Base Preprocessing Pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('feat_eng', FunctionTransformer(feature_engineering)),
    ('preprocessor', preprocessor)
])

# ==========================================
# 3. MODEL COMPARISON (STEP 5)
# ==========================================

models_to_train = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []

print("Comparing Models...")
for name, model in models_to_train.items():
    current_pipe = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', model)
    ])
    
    current_pipe.fit(X_train, y_train)
    y_pred = current_pipe.predict(X_test)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, current_pipe.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else 0
    })

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
print("\nModel Comparison Table:")
print(results_df)

# Automatically Select Best Model
best_model_name = results_df.iloc[0]['Model']
best_model_obj = models_to_train[best_model_name]
print(f"\nWinning Model: {best_model_name}")

# ==========================================
# 4. CROSS-VALIDATION & TUNING (STEP 6 & 7)
# ==========================================

# Create final pipeline with winner
final_pipe = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('model', best_model_obj)
])

# Cross Validation
cv_scores = cross_val_score(final_pipe, X_train, y_train, cv=5)
print(f"\nCross-Validation Mean Accuracy: {cv_scores.mean():.4f}")

# Hyperparameter Tuning (Grid Search)
# Note: Grid defined for Gradient Boosting as it is the likely winner
param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.1],
    'model__max_depth': [3, 5]
}

print("\nStarting Hyperparameter Tuning...")
grid_search = GridSearchCV(final_pipe, param_grid, cv=5, scoring='accuracy', verbose=0)
grid_search.fit(X_train, y_train)

# ==========================================
# 5. FINAL EVALUATION (STEP 8 & 9)
# ==========================================

best_model = grid_search.best_estimator_
y_final_pred = best_model.predict(X_test)

print("\nBest Parameters Found:", grid_search.best_params_)
print("\nFinal Classification Report:")
print(classification_report(y_test, y_final_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_final_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix: {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the model
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
print("\nModel saved as best_model.pkl")