import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 한글 폰트 설정 (Windows용 예시: 'Malgun Gothic')
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv('풍력/data/풍력_전처리_이상치제거.csv')

# 사용할 피처 및 타겟
feature_cols = [
    '설비용량(kW)', '연식(년)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '적설(cm)', '태양고도',
    '발전구분', '지점명', '시간', '하늘상태'
]
target_col = '발전량(kWh)'

# 결측치 제거
df = df.dropna(subset=feature_cols + [target_col])

# 범주형 변수 지정
cat_features = ['발전구분', '지점명', '시간', '풍향(16방위)', '하늘상태']


# cat_features 컬럼을 문자열로 변환 (CatBoost 오류 방지)
for col in cat_features:
    df[col] = df[col].astype(str)

# train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], df[target_col], test_size=0.2, random_state=42
)

# CatBoost Pool 생성
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# CatBoost 모델 학습
model = CatBoostRegressor(verbose=100, random_state=42, task_type='GPU')
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)


# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}')

# --- KFold 교차검증 (CatBoost) ---
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_list = []
for train_idx, val_idx in kf.split(df[feature_cols]):
    X_tr, X_val = df.iloc[train_idx][feature_cols], df.iloc[val_idx][feature_cols]
    y_tr, y_val = df.iloc[train_idx][target_col], df.iloc[val_idx][target_col]
    # cat_features 인덱스 전달
    cat_idx = [X_tr.columns.get_loc(col) for col in cat_features]
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_idx)
    pool_val = Pool(X_val, y_val, cat_features=cat_idx)
    model_cv = CatBoostRegressor(verbose=0, random_state=42, task_type='GPU')
    model_cv.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=30)
    y_val_pred = model_cv.predict(X_val)
    rmse_cv = np.sqrt(mean_squared_error(y_val, y_val_pred))
    cv_rmse_list.append(rmse_cv)
print(f'CatBoost 5-Fold CV RMSE: {np.mean(cv_rmse_list):.2f} ± {np.std(cv_rmse_list):.2f}')

# 피처 중요도 시각화
importances = model.get_feature_importance()
plt.figure(figsize=(8,5))
plt.barh(feature_cols, importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.show()

# 상위 N개 피처만 사용해서 재학습 (예: 상위 5개)
N = 5
top_features = [feature for _, feature in sorted(zip(importances, feature_cols), reverse=True)][:N]
print('Top features:', top_features)

# top_features 중 cat_features에 해당하는 것만 추출
top_cat_features = [col for col in top_features if col in cat_features]

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]
train_pool_top = Pool(X_train_top, y_train, cat_features=top_cat_features)
test_pool_top = Pool(X_test_top, y_test, cat_features=top_cat_features)

model_top = CatBoostRegressor(verbose=100, random_state=42, task_type='GPU')
model_top.fit(train_pool_top, eval_set=test_pool_top, early_stopping_rounds=50)
y_pred_top = model_top.predict(X_test_top)
mse_top = mean_squared_error(y_test, y_pred_top)
rmse_top = np.sqrt(mse_top)
print(f'RMSE (Top {N} features): {rmse_top:.2f}')