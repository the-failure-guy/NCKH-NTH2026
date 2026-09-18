import joblib, pandas as pd
from sbase_pipeline import find_and_load_nhanes_data, preprocess_and_feature_engineering
df_mcq, df_demo = find_and_load_nhanes_data('./Training/Model/MedicalCondition')
df = preprocess_and_feature_engineering(df_mcq, df_demo)
art = joblib.load('./Training/Model/MedicalCondition/sbase_model.joblib')
features = art['features']
df[features] = df[features].fillna(df[features].median())
healthy = df[df['S_base'] == 0.0][features].iloc[0].to_dict()
sick = df[df['S_base'] == 1.0][features].iloc[0].to_dict()
joblib.dump({'healthy': healthy, 'sick': sick}, './Training/mock_patients.joblib')
print("Dumped mock patients.")
