import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

# 데이터 불러오기
df = pd.read_csv('풍력/data/풍력_전처리_이상치제거.csv')

# 사용할 피처 및 타겟
feature_cols = [
    '설비용량(kW)', '연식(년)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '적설(cm)', '태양고도',
    '발전구분', '지점명', '시간', '하늘상태'
]
target_col = '발전량(kWh)'

# '일자', '호기' 등은 feature에서 제외
df = df[feature_cols + [target_col]].dropna()

cat_features = ['발전구분', '지점명', '시간', '풍향(16방위)', '하늘상태']

# 범주형 변수 원핫인코딩
df_encoded = pd.get_dummies(df, columns=cat_features)

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KFold 객체 생성
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'RandomForest RMSE: {rmse_rf:.2f}')
rf_scores = cross_val_score(rf, X, y, cv=kf, scoring='neg_root_mean_squared_error')
print(f'RandomForest 5-Fold CV RMSE: {-rf_scores.mean():.2f} ± {rf_scores.std():.2f}')


# 2. XGBoost (GPU 사용, 최신 권장 방식)
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=100, random_state=42, tree_method='hist', device='cuda')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'XGBoost RMSE: {rmse_xgb:.2f}')
xgb_scores = cross_val_score(
    XGBRegressor(n_estimators=100, random_state=42, tree_method='hist', device='cuda'),
    X, y, cv=kf, scoring='neg_root_mean_squared_error')
print(f'XGBoost 5-Fold CV RMSE: {-xgb_scores.mean():.2f} ± {xgb_scores.std():.2f}')

# 3. LightGBM (GPU 사용)
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(n_estimators=100, random_state=42, device='gpu')
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print(f'LightGBM RMSE: {rmse_lgbm:.2f}')
lgbm_scores = cross_val_score(lgbm, X, y, cv=kf, scoring='neg_root_mean_squared_error')
print(f'LightGBM 5-Fold CV RMSE: {-lgbm_scores.mean():.2f} ± {lgbm_scores.std():.2f}')

# 4. LinearRegression (baseline)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f'LinearRegression RMSE: {rmse_lr:.2f}')
lr_scores = cross_val_score(lr, X, y, cv=kf, scoring='neg_root_mean_squared_error')
print(f'LinearRegression 5-Fold CV RMSE: {-lr_scores.mean():.2f} ± {lr_scores.std():.2f}')