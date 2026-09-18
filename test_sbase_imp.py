import joblib, numpy as np
art = joblib.load('./Training/Model/MedicalCondition/sbase_model.joblib')
xgb, features = art['model'], art['features']
imp = xgb.feature_importances_
idx = np.argsort(imp)[-10:]
for i in idx: print(f"{features[i]}: {imp[i]}")
