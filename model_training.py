import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_water_quality_model():
    # 1. Simulate or load dataset (using standard water-quality parameters layout)
    # Expected features: ph, turbidity, temperature, dissolved_oxygen, conductivity
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'ph': np.random.normal(7.0, 1.2, n_samples),
        'turbidity': np.random.normal(3.5, 1.5, n_samples),
        'temperature': np.random.normal(22.0, 4.0, n_samples),
        'dissolved_oxygen': np.random.normal(7.5, 2.0, n_samples),
        'conductivity': np.random.normal(400, 100, n_samples),
    })
    
    # Inject synthetic missing values to simulate real-world data quality checks (Day 3 requirement)
    data.loc[np.random.choice(n_samples, 20), 'ph'] = np.nan
    data.loc[np.random.choice(n_samples, 15), 'turbidity'] = np.nan

    # Target logic: Risk Category (0: Safe/Low Risk, 1: High Risk/Polluted)
    conditions = (
        (data['ph'].between(6.5, 8.5)) & 
        (data['turbidity'] < 5.0) & 
        (data['dissolved_oxygen'] > 6.0)
    )
    data['risk_category'] = np.where(conditions, 0, 1)
    
    X = data.drop(columns=['risk_category'])
    y = data['risk_category']
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Preprocessing: Imputation + Scaling
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # 4. Model Training (Foundational Decision Tree Classifier)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 5. Evaluation & Metrics Interpretation (Day 5 requirement)
    y_pred = clf.predict(X_test_scaled)
    print("--- MODEL EVALUATION METRICS ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # 6. Save Artifacts for Day 7 Streamlit App
    joblib.dump(clf, 'water_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    print("\n[INFO] Model training complete. Artifacts saved successfully.")

if __name__ == "__main__":
    train_water_quality_model()