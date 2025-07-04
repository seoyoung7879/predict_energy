import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# 평가 지표 함수
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# 데이터 불러오기
df = pd.read_csv('풍력/data/풍력_전처리_이상치제거.csv')

feature_cols = [
    '설비용량(kW)', '연식(년)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)',
    '적설(cm)', '태양고도', '발전구분', '지점명', '시간', '하늘상태'
]
target_col = '발전량(kWh)'

df = df.dropna(subset=feature_cols + [target_col])
cat_features = ['발전구분', '지점명', '시간', '풍향(16방위)', '하늘상태']
for col in cat_features:
    df[col] = df[col].astype(str)

# 날짜 파생변수 추가 (예시)
if '일자' in df.columns:
    df['일자'] = pd.to_datetime(df['일자'])
    df['월'] = df['일자'].dt.month
    df['계절'] = df['월'] % 12 // 3 + 1
    df['요일'] = df['일자'].dt.weekday
    feature_cols += ['월', '계절', '요일']

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_rf_train = pd.get_dummies(X_train, columns=cat_features)
X_rf_test = pd.get_dummies(X_test, columns=cat_features)
X_rf_train, X_rf_test = X_rf_train.align(X_rf_test, join='left', axis=1, fill_value=0)

# --- CatBoost 하이퍼파라미터 튜닝 ---
cat_model = CatBoostRegressor(
    verbose=0,
    random_state=42,
    task_type='CPU',
    cat_features=cat_features
)

cat_params = {
    'depth': [4, 6, 8],               # 범위 축소
    'learning_rate': [0.03, 0.1],    # 범위 축소
    'iterations': [300, 500]          # 범위 축소
}

cat_search = RandomizedSearchCV(
    cat_model,
    cat_params,
    n_iter=5,           # 탐색 횟수 줄임
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,          # CPU 병렬 사용
    verbose=2,
    random_state=42
)

cat_search.fit(X_train, y_train)
print('Best CatBoost params:', cat_search.best_params_)

# 최적 파라미터로 GPU 학습
cat_model_best = CatBoostRegressor(
    **cat_search.best_params_,
    verbose=2,
    random_state=42,
    task_type='GPU',
    cat_features=cat_features
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

cat_model_best.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

y_pred_cat = cat_model_best.predict(X_test)

# --- RandomForest 하이퍼파라미터 튜닝 (RandomizedSearchCV) ---
rf_model = RandomForestRegressor(random_state=42)

rf_params = {
    'n_estimators': [100, 200, 300],  # 살짝 확장
    'max_depth': [None, 10, 20]
}

rf_search = RandomizedSearchCV(
    rf_model,
    rf_params,
    n_iter=5,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

rf_search.fit(X_rf_train, y_train)
print("Best RF params:", rf_search.best_params_)

rf_model_best = rf_search.best_estimator_
y_pred_rf = rf_model_best.predict(X_rf_test)

# --- 앙상블 예측 및 평가 ---
y_pred_ensemble = (y_pred_cat + y_pred_rf) / 2

r2_ensemble = r2_score(y_test, y_pred_ensemble)
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
medae_ensemble = median_absolute_error(y_test, y_pred_ensemble)
smape_ensemble = smape(y_test.values, y_pred_ensemble)

print(f'앙상블(RF+CatBoost) R²: {r2_ensemble:.4f}')
print(f'앙상블(RF+CatBoost) MAE: {mae_ensemble:.2f}')
print(f'앙상블(RF+CatBoost) MedianAE: {medae_ensemble:.2f}')
print(f'앙상블(RF+CatBoost) sMAPE: {smape_ensemble:.2f}%')

# --- ShuffleSplit 교차검증 (빠른 검증) ---
from sklearn.model_selection import ShuffleSplit
n_splits = 3
ss = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

cv_mae_list = []
cv_medae_list = []
cv_smape_list = []
cv_r2_list = []

for train_idx, val_idx in ss.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # CatBoost (GPU, iteration 300)
    cat_idx = [X_tr.columns.get_loc(col) for col in cat_features]
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_idx)
    pool_val = Pool(X_val, y_val, cat_features=cat_idx)
    model_cb = CatBoostRegressor(verbose=0, random_state=42, task_type='GPU', iterations=300)
    model_cb.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=20)
    y_val_pred_cb = model_cb.predict(X_val)

    # RandomForest (원핫인코딩)
    X_tr_rf = pd.get_dummies(X_tr, columns=cat_features)
    X_val_rf = pd.get_dummies(X_val, columns=cat_features)
    X_tr_rf, X_val_rf = X_tr_rf.align(X_val_rf, join='left', axis=1, fill_value=0)
    rf_cv = RandomForestRegressor(n_estimators=rf_model_best.n_estimators, max_depth=rf_model_best.max_depth, random_state=42)
    rf_cv.fit(X_tr_rf, y_tr)
    y_val_pred_rf = rf_cv.predict(X_val_rf)

    # 앙상블 예측
    y_val_pred_ensemble = (y_val_pred_cb + y_val_pred_rf) / 2

    # 지표 계산
    cv_mae_list.append(mean_absolute_error(y_val, y_val_pred_ensemble))
    cv_medae_list.append(median_absolute_error(y_val, y_val_pred_ensemble))
    cv_smape_list.append(smape(y_val.values, y_val_pred_ensemble))
    cv_r2_list.append(r2_score(y_val, y_val_pred_ensemble))

print(f'앙상블(RF+CatBoost) ShuffleSplit MAE: {np.mean(cv_mae_list):.2f} ± {np.std(cv_mae_list):.2f}')
print(f'앙상블(RF+CatBoost) ShuffleSplit MedianAE: {np.mean(cv_medae_list):.2f} ± {np.std(cv_medae_list):.2f}')
print(f'앙상블(RF+CatBoost) ShuffleSplit sMAPE: {np.mean(cv_smape_list):.2f}% ± {np.std(cv_smape_list):.2f}%')
print(f'앙상블(RF+CatBoost) ShuffleSplit R²: {np.mean(cv_r2_list):.4f} ± {np.std(cv_r2_list):.4f}')
