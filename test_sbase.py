import joblib, numpy as np, pandas as pd
art = joblib.load('./Training/Model/MedicalCondition/sbase_model.joblib')
xgb, features = art['model'], art['features']
# Healthy mock
patient = {f: 2.0 for f in features}
input_df = pd.DataFrame([patient], columns=features)
prob = xgb.predict_proba(input_df)[0][1]
print(f"Prob Healthy: {prob:.4f}")

# Sick mock
patient_sick = {f: 1.0 for f in features}
input_sick = pd.DataFrame([patient_sick], columns=features)
prob_sick = xgb.predict_proba(input_sick)[0][1]
print(f"Prob Sick: {prob_sick:.4f}")
