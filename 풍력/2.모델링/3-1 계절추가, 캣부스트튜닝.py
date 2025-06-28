import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
df = pd.read_csv('풍력/data/풍력_전처리_이상치제거.csv')

feature_cols = [
    '설비용량(kW)', '연식(년)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '적설(cm)', '태양고도',
    '발전구분', '지점명', '시간', '하늘상태'
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

# CatBoost 모델 초기화 시 cat_features 지정
cat_model = CatBoostRegressor(
    verbose=2, 
    random_state=42, 
    task_type='GPU',
    cat_features=cat_features
)

# 하이퍼파라미터 탐색용 파라미터 그리드
cat_params = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.1],
    'iterations': [300, 500, 1000]
}

# RandomizedSearchCV는 cat_features 없이 fit함
cat_search = RandomizedSearchCV(
    cat_model, cat_params, n_iter=10, cv=3, 
    scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2, random_state=42
)

cat_search.fit(X_train, y_train)

print('Best CatBoost params:', cat_search.best_params_)

cat_model_best = cat_search.best_estimator_

# Pool 만들어서 실제 학습과 평가 (early stopping 포함)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

cat_model_best.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

y_pred_cat = cat_model_best.predict(X_test)

# RandomForest 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

rf_grid.fit(X_rf_train, y_train)
print("Best RF params:", rf_grid.best_params_)

rf_model = rf_grid.best_estimator_
y_pred_rf = rf_model.predict(X_rf_test)

# 앙상블(단순 평균)
y_pred_ensemble = (y_pred_cat + y_pred_rf) / 2
rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
print(f'앙상블(RF+CatBoost) RMSE: {rmse_ensemble:.2f}')

# --- 5-Fold 교차검증 (앙상블) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ensemble_rmse_list = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # CatBoost
    cat_idx = [X_tr.columns.get_loc(col) for col in cat_features]
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_idx)
    pool_val = Pool(X_val, y_val, cat_features=cat_idx)
    model_cb = CatBoostRegressor(verbose=0, random_state=42, task_type='GPU')
    model_cb.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=30)
    y_val_pred_cb = model_cb.predict(X_val)
    # RandomForest (원핫인코딩)
    X_tr_rf = pd.get_dummies(X_tr, columns=cat_features)
    X_val_rf = pd.get_dummies(X_val, columns=cat_features)
    # 컬럼 맞추기
    X_tr_rf, X_val_rf = X_tr_rf.align(X_val_rf, join='left', axis=1, fill_value=0)
    rf_cv = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_cv.fit(X_tr_rf, y_tr)
    y_val_pred_rf = rf_cv.predict(X_val_rf)
    # 앙상블
    y_val_pred_ensemble = (y_val_pred_cb + y_val_pred_rf) / 2
    rmse_ens = np.sqrt(mean_squared_error(y_val, y_val_pred_ensemble))
    ensemble_rmse_list.append(rmse_ens)
print(f'앙상블(RF+CatBoost) 5-Fold CV RMSE: {np.mean(ensemble_rmse_list):.2f} ± {np.std(ensemble_rmse_list):.2f}')