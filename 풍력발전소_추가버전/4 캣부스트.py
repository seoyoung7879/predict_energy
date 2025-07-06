import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 풍력발전량 예측 모델 (CatBoost) ===")

# 1. 데이터 로드
print("\n1. 데이터 로딩...")
df = pd.read_csv('전처리완료_기상과풍력.csv')
print(f"데이터 형태: {df.shape}")
print(f"컬럼: {list(df.columns)}")

# 2. 데이터 전처리
print("\n2. 데이터 전처리...")

# 범주형 변수 인코딩
categorical_features = ['발전구분', '계절']
label_encoders = {}

for feature in categorical_features:
    if feature in df.columns:
        le = LabelEncoder()
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])
        label_encoders[feature] = le
        print(f"   {feature} 인코딩 완료: {len(le.classes_)}개 클래스")

# 결측값 확인
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    print(f"   결측값 발견: {missing_data[missing_data > 0]}")
    df = df.fillna(0)
else:
    print("   결측값 없음")

# 3. 특성 및 타겟 변수 설정
print("\n3. 특성 및 타겟 변수 설정...")

# 타겟 변수
target = '발전량(kWh)'

# 특성 변수 (수치형 + 인코딩된 범주형)
numeric_features = [
    '호기', '월', '설비용량(MW)', '연식(년)', 
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 
    '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', 
    '일조(hr)', '일사(MJ/m2)', '적설(cm)', 
    '풍향_sin', '시간_sin', '블레이드', '정격', '커트인', '커트아웃'
]

categorical_encoded_features = [f'{cat}_encoded' for cat in categorical_features if f'{cat}_encoded' in df.columns]
features = numeric_features + categorical_encoded_features

# 실제 존재하는 컬럼만 선택
available_features = [col for col in features if col in df.columns]
X = df[available_features]
y = df[target]

print(f"   특성 변수: {len(available_features)}개")
print(f"   타겟 변수: {target}")
print(f"   데이터 포인트: {len(X):,}개")

# 4. 데이터 분할
print("\n4. 데이터 분할...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['발전구분']
)

print(f"   훈련 데이터: {X_train.shape[0]:,}개")
print(f"   테스트 데이터: {X_test.shape[0]:,}개")

# 5. 기본 CatBoost 모델 훈련
print("\n5. 기본 CatBoost 모델 훈련...")
base_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=False
)

base_model.fit(X_train, y_train)

# 기본 모델 예측 및 평가
y_pred_base = base_model.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
r2_base = r2_score(y_test, y_pred_base)
mae_base = mean_absolute_error(y_test, y_pred_base)
median_error_base = np.median(np.abs(y_test - y_pred_base))

print(f"   기본 모델 성능:")
print(f"     RMSE: {rmse_base:.2f}")
print(f"     R²: {r2_base:.4f}")
print(f"     MAE: {mae_base:.2f}")
print(f"     오차 중앙값: {median_error_base:.2f}")

# 6. 하이퍼파라미터 튜닝
print("\n6. 하이퍼파라미터 튜닝...")

# 적당한 수준의 파라미터 그리드
param_grid = {
    'iterations': [800, 1000, 1200],
    'learning_rate': [0.05, 0.1, 0.15],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5]
}

print(f"   그리드 서치 진행 중... (총 {np.prod([len(v) for v in param_grid.values()])}개 조합)")

# GridSearchCV 실행
catboost_model = CatBoostRegressor(random_seed=42, verbose=False)
grid_search = GridSearchCV(
    catboost_model, 
    param_grid, 
    cv=3,  # 3-fold CV로 시간 단축
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"   최적 파라미터: {grid_search.best_params_}")
print(f"   최적 CV 점수: {-grid_search.best_score_:.2f}")

# 7. 최적 모델 예측 및 평가
print("\n7. 최적 모델 평가...")
y_pred_best = best_model.predict(X_test)

rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
median_error_best = np.median(np.abs(y_test - y_pred_best))

print(f"   최적 모델 성능:")
print(f"     RMSE: {rmse_best:.2f}")
print(f"     R²: {r2_best:.4f}")
print(f"     MAE: {mae_best:.2f}")
print(f"     오차 중앙값: {median_error_best:.2f}")

# 성능 개선도 계산
rmse_improvement = ((rmse_base - rmse_best) / rmse_base) * 100
r2_improvement = ((r2_best - r2_base) / r2_base) * 100

print(f"\n   성능 개선:")
print(f"     RMSE 개선: {rmse_improvement:.1f}%")
print(f"     R² 개선: {r2_improvement:.1f}%")

# 8. Feature Importance 분석
print("\n8. Feature Importance 분석...")
feature_importance = best_model.feature_importances_
feature_names = X_train.columns

# Feature importance 데이터프레임 생성
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("   상위 10개 중요 특성:")
for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"     {i:2d}. {row['feature']}: {row['importance']:.4f}")

# 9. 시각화
print("\n9. 결과 시각화...")

# 9-1. 예측 vs 실제 산점도
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred_best, alpha=0.5, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 발전량 (kWh)')
plt.ylabel('예측 발전량 (kWh)')
plt.title(f'예측 vs 실제\nR² = {r2_best:.4f}')
plt.grid(True, alpha=0.3)

# 9-2. 잔차 플롯
plt.subplot(2, 3, 2)
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, alpha=0.5, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('예측 발전량 (kWh)')
plt.ylabel('잔차 (실제 - 예측)')
plt.title('잔차 플롯')
plt.grid(True, alpha=0.3)

# 9-3. 오차 히스토그램
plt.subplot(2, 3, 3)
plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('잔차 (kWh)')
plt.ylabel('빈도')
plt.title(f'오차 분포\n중앙값: {median_error_best:.2f}')
plt.grid(True, alpha=0.3)

# 9-4. Feature Importance (상위 15개)
plt.subplot(2, 3, 4)
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('중요도')
plt.title('Feature Importance (상위 15개)')
plt.gca().invert_yaxis()

# 9-5. 실제 vs 예측 곡선 (시간순 샘플)
plt.subplot(2, 3, 5)
# 시간순으로 정렬된 샘플 일부 선택
sample_size = min(500, len(y_test))
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
sample_idx = sorted(sample_idx)

plt.plot(y_test.iloc[sample_idx].values, label='실제', alpha=0.7, linewidth=1)
plt.plot(y_pred_best[sample_idx], label='예측', alpha=0.7, linewidth=1)
plt.xlabel('샘플 순서')
plt.ylabel('발전량 (kWh)')
plt.title('실제 vs 예측 곡선 비교')
plt.legend()
plt.grid(True, alpha=0.3)

# 9-6. 발전구분별 성능
plt.subplot(2, 3, 6)
test_df = X_test.copy()
test_df['실제'] = y_test.values
test_df['예측'] = y_pred_best
test_df['발전구분'] = df.loc[y_test.index, '발전구분'].values

plant_performance = []
for plant in test_df['발전구분'].unique():
    plant_data = test_df[test_df['발전구분'] == plant]
    plant_r2 = r2_score(plant_data['실제'], plant_data['예측'])
    plant_performance.append({'발전구분': plant, 'R²': plant_r2})

plant_perf_df = pd.DataFrame(plant_performance).sort_values('R²', ascending=True)
plt.barh(range(len(plant_perf_df)), plant_perf_df['R²'])
plt.yticks(range(len(plant_perf_df)), plant_perf_df['발전구분'])
plt.xlabel('R² 점수')
plt.title('발전구분별 모델 성능')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# 10. 발전구분별 상세 성능 분석
print("\n10. 발전구분별 상세 성능 분석...")
print(f"{'발전구분':<10} {'데이터수':<8} {'RMSE':<8} {'R²':<8} {'MAE':<8}")
print("-" * 50)

for plant in test_df['발전구분'].unique():
    plant_data = test_df[test_df['발전구분'] == plant]
    plant_rmse = np.sqrt(mean_squared_error(plant_data['실제'], plant_data['예측']))
    plant_r2 = r2_score(plant_data['실제'], plant_data['예측'])
    plant_mae = mean_absolute_error(plant_data['실제'], plant_data['예측'])
    
    print(f"{plant:<10} {len(plant_data):<8} {plant_rmse:<8.2f} {plant_r2:<8.4f} {plant_mae:<8.2f}")

# 11. 모델 저장 (선택사항)
print("\n11. 모델 및 결과 저장...")

# 예측 결과 저장
results_df = pd.DataFrame({
    '실제_발전량': y_test.values,
    '예측_발전량': y_pred_best,
    '오차': y_test.values - y_pred_best,
    '절대오차': np.abs(y_test.values - y_pred_best)
})
results_df.to_csv('catboost_예측결과.csv', index=False, encoding='utf-8-sig')

# Feature importance 저장
importance_df.to_csv('catboost_feature_importance.csv', index=False, encoding='utf-8-sig')

print(f"   예측 결과 저장: catboost_예측결과.csv")
print(f"   Feature importance 저장: catboost_feature_importance.csv")

# 12. 최종 요약
print("\n" + "="*60)
print("                    최종 모델 성능 요약")
print("="*60)
print(f"RMSE (평균제곱근오차):     {rmse_best:.2f} kWh")
print(f"R² (결정계수):            {r2_best:.4f}")
print(f"MAE (평균절대오차):        {mae_best:.2f} kWh")
print(f"오차 중앙값:              {median_error_best:.2f} kWh")
print(f"훈련 데이터:              {len(X_train):,}개")
print(f"테스트 데이터:            {len(X_test):,}개")
print(f"사용된 특성:              {len(available_features)}개")
print("="*60)
