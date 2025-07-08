import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 풍력발전량 예측 모델 (CatBoost, 파생변수 포함, 하이퍼파라미터 튜닝) ===")

# 1. 데이터 로드
print("\n1. 데이터 로딩...")
df = pd.read_csv('파생변수추가_기상과풍력.csv')
print(f"데이터 형태: {df.shape}")
print(f"컬럼: {list(df.columns)}")

# 2. 데이터 전처리
print("\n2. 데이터 전처리...")

# 범주형 변수 인코딩
categorical_features = ['계절']
label_encoders = {}
for feature in categorical_features:
    if feature in df.columns:
        le = LabelEncoder()
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])
        label_encoders[feature] = le
        print(f"   {feature} 인코딩 완료: {len(le.classes_)}개 클래스")

# 결측값 처리
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    print(f"   결측값 발견: {missing_data[missing_data > 0]}")
    df = df.fillna(0)
else:
    print("   결측값 없음")

print("\n[INFO] 데이터 준비 및 전처리 완료.")

# 3. 특성 및 타겟 변수 설정
print("\n3. 특성 및 타겟 변수 설정...")

target = '발전량(kWh)'
features = [
    '설비용량(MW)', '연식(년)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '적설(cm)',
    '풍향_sin', '풍향_cos', '시간_sin', '시간_cos', '월_sin', '월_cos',
    '블레이드', '정격', '커트인', '커트아웃', 'air_density', 'absolute_humidity',
    '계절_encoded'
]
available_features = [col for col in features if col in df.columns]
X = df[available_features]
y = df[target]

print(f"   특성 변수: {len(available_features)}개")
print(f"   타겟 변수: {target}")
print(f"   데이터 포인트: {len(X):,}개")

# 4. 데이터 분할
print("\n4. 데이터 분할...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   훈련 데이터: {X_train.shape[0]:,}개")
print(f"   테스트 데이터: {X_test.shape[0]:,}개")

print("\n[INFO] 학습/테스트 데이터 분할 완료.")

# 5. 하이퍼파라미터 튜닝 (조금 더 자세히)
print("\n5. 하이퍼파라미터 튜닝(GridSearchCV, CPU, 확장)...")
param_grid = {
    'iterations': [400, 600, 800, 1000],
    'learning_rate': [0.05, 0.07, 0.1],
    'depth': [6, 8, 10],
    'l2_leaf_reg': [1, 3, 5],
    'bootstrap_type': ['Bayesian', 'Bernoulli']
}
catboost_model = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0)
grid_search = GridSearchCV(
    catboost_model,
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"   최적 파라미터: {best_params}")
print(f"   최적 CV 점수: {-grid_search.best_score_:.2f}")

print("\n[INFO] 하이퍼파라미터 튜닝 완료. 최적 파라미터로 GPU 재학습 시작...")

# 6. 최적 모델(GPU)로 재학습 및 평가
print("\n6. 최적 파라미터로 GPU CatBoost 재학습...")
best_model = CatBoostRegressor(
    **best_params,
    loss_function='RMSE',
    random_seed=42,
    verbose=0,
    task_type='GPU',
    devices='0'
)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
median_error = np.median(np.abs(y_test - y_pred))

print(f"   모델 성능:")
print(f"     RMSE: {rmse:.2f}")
print(f"     R²: {r2:.4f}")
print(f"     MAE: {mae:.2f}")
print(f"     오차 중앙값: {median_error:.2f}")

# Feature Importance
importance = best_model.feature_importances_
importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importance}).sort_values('importance', ascending=False)
print("\n상위 10개 중요 특성:")
for i, (_, row) in enumerate(importance_df.head(23).iterrows(), 1):
    print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")

# Feature Importance 전체 출력
print("\n[INFO] 전체 Feature Importance 출력")
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")

# 예측 vs 실제 곡선 그래프
print("\n[INFO] 예측 vs 실제 곡선 시각화")
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:300], label='실제', alpha=0.7)
plt.plot(y_pred[:300], label='예측', alpha=0.7)
plt.xlabel('샘플 순서')
plt.ylabel('발전량 (kWh)')
plt.title('실제 vs 예측 곡선')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Feature importance 시각화
print("\n[INFO] Feature Importance 시각화")
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('중요도')
plt.title('Feature Importance (상위 15개)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 예측/실제 KDE 그래프
print("\n[INFO] 전체 KDE 곡선 시각화")
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='실제', color='blue', fill=True, alpha=0.4)
sns.kdeplot(y_pred, label='예측', color='orange', fill=True, alpha=0.4)
plt.xlabel('발전량 (kWh)')
plt.ylabel('밀도')
plt.title('실제 vs 예측 발전량 KDE 곡선')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 발전구분별 상세 성능 분석
print("\n발전구분별 상세 성능 분석:")
if '발전구분' in df.columns:
    test_df = X_test.copy()
    test_df['실제'] = y_test.values
    test_df['예측'] = y_pred
    test_df['발전구분'] = df.loc[y_test.index, '발전구분'].values
    print(f"{'발전구분':<10} {'N':<8} {'RMSE':<8} {'R²':<8} {'MAE':<8} {'오차중앙값':<10}")
    print("-"*55)
    for plant in test_df['발전구분'].unique():
        plant_data = test_df[test_df['발전구분'] == plant]
        plant_rmse = np.sqrt(mean_squared_error(plant_data['실제'], plant_data['예측']))
        plant_r2 = r2_score(plant_data['실제'], plant_data['예측'])
        plant_mae = mean_absolute_error(plant_data['실제'], plant_data['예측'])
        plant_median = np.median(np.abs(plant_data['실제'] - plant_data['예측']))
        print(f"{plant:<10} {len(plant_data):<8} {plant_rmse:<8.2f} {plant_r2:<8.4f} {plant_mae:<8.2f} {plant_median:<10.2f}")
    # 발전구분별 KDE 그래프
    plt.figure(figsize=(12, 6))
    for plant in test_df['발전구분'].unique():
        plant_data = test_df[test_df['발전구분'] == plant]
        sns.kdeplot(plant_data['실제'], label=f'{plant} 실제', fill=True, alpha=0.2)
        sns.kdeplot(plant_data['예측'], label=f'{plant} 예측', fill=True, alpha=0.2, linestyle='--')
    plt.xlabel('발전량 (kWh)')
    plt.ylabel('밀도')
    plt.title('발전구분별 실제/예측 발전량 KDE 곡선')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 오차가 큰 케이스(상위 10%) 저장 및 분석
print("\n[INFO] 오차 상위 10% 케이스 저장 및 분석...")
results_df = X_test.copy()
results_df['실제'] = y_test.values
results_df['예측'] = y_pred
results_df['오차'] = y_test.values - y_pred
results_df['절대오차'] = np.abs(y_test.values - y_pred)
results_df['발전구분'] = df.loc[y_test.index, '발전구분'].values if '발전구분' in df.columns else 'N/A'
# 상위 10% 추출
n_top = int(len(results_df) * 0.10)
results_df_sorted = results_df.sort_values('절대오차', ascending=False)
top_error_df = results_df_sorted.head(n_top)
top_error_df.to_csv('catboost_오차상위10p.csv', index=False, encoding='utf-8-sig')
print(f"   절대오차 상위 10% ({n_top}건) 저장: catboost_오차상위10p.csv")

# 발전구분별 평가지표 csv 저장
if '발전구분' in df.columns:
    metrics = []
    for plant in results_df['발전구분'].unique():
        plant_data = results_df[results_df['발전구분'] == plant]
        plant_rmse = np.sqrt(mean_squared_error(plant_data['실제'], plant_data['예측']))
        plant_r2 = r2_score(plant_data['실제'], plant_data['예측'])
        plant_mae = mean_absolute_error(plant_data['실제'], plant_data['예측'])
        plant_median = np.median(np.abs(plant_data['실제'] - plant_data['예측']))
        metrics.append({
            '발전구분': plant,
            'N': len(plant_data),
            'RMSE': plant_rmse,
            'R2': plant_r2,
            'MAE': plant_mae,
            '오차중앙값': plant_median
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('catboost_발전구분별_평가지표.csv', index=False, encoding='utf-8-sig')
    print("\n[INFO] 발전구분별 평가지표:")
    print(metrics_df)
    print("   csv 저장: catboost_발전구분별_평가지표.csv")
