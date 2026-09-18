import joblib, numpy as np, pandas as pd
from samap_pipeline import calculate_albi, calculate_amap_raw
art = joblib.load('./Training/samap_model.joblib')
xgb, scaler = art['model'], art['scaler']
age, gender, alb, bili, plt = 55, 1, 42, 12, 250
albi = calculate_albi(bili, alb)
amap = calculate_amap_raw(age, gender, albi, plt)
s_amap = scaler.transform(np.array([[amap]]))[0][0]
input_df = pd.DataFrame([{'age': age, 'gender': gender, 'albumin': alb, 'bilirubin': bili, 'platelets': plt, 'ALBI_score': albi, 'S_aMAP': s_amap}])
prob = xgb.predict_proba(input_df)[0][1]
print(f"Prob: {prob:.4f}")
