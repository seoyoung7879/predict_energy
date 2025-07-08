import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 풍력발전량 예측 모델 (RandomForest, 파생변수 포함) ===")

# 데이터 로드
print("\n1. 데이터 로딩...")
df = pd.read_csv('파생변수추가_기상과풍력.csv')
print(f"데이터 형태: {df.shape}")

# 범주형 변수 인코딩
categorical_features = ['계절']
for feature in categorical_features:
    if feature in df.columns:
        le = LabelEncoder()
        df[f'{feature}_encoded'] = le.fit_transform(df[feature])

# 특성 및 타겟 변수
features = [
    '설비용량(MW)', '연식(년)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
    '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '적설(cm)',
    '풍향_sin', '풍향_cos', '시간_sin', '시간_cos', '월_sin', '월_cos',
    '블레이드', '정격', '커트인', '커트아웃', 'air_density', 'absolute_humidity',
    '계절_encoded'
]
features = [col for col in features if col in df.columns]
X = df[features]
y = df['발전량(kWh)']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   훈련 데이터: {X_train.shape[0]:,}개, 테스트 데이터: {X_test.shape[0]:,}개")

# 하이퍼파라미터 튜닝 (간소화)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 16],
    'min_samples_split': [2, 5]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
gs = GridSearchCV(rf, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
gs.fit(X_train, y_train)
print(f"최적 파라미터: {gs.best_params_}")

# 최적 모델로 예측 및 평가
best_rf = gs.best_estimator_
y_pred = best_rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
median_error = np.median(np.abs(y_test - y_pred))

print(f"\n모델 성능:")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²: {r2:.4f}")
print(f"  MAE: {mae:.2f}")
print(f"  오차 중앙값: {median_error:.2f}")

# 예측 결과 샘플 출력
print("\n예측 결과 샘플 (실제 vs 예측 vs 오차):")
for i in range(10):
    print(f"{i+1:2d}. 실제: {y_test.values[i]:.2f} | 예측: {y_pred[i]:.2f} | 오차: {y_test.values[i] - y_pred[i]:.2f}")

# Feature Importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(np.array(X_train.columns)[indices][:23], importances[indices][:23])
plt.xlabel('중요도')
plt.title('Feature Importance (상위 15개)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 예측/실제 KDE 그래프
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='실제', color='blue', fill=True, alpha=0.4)
sns.kdeplot(y_pred, label='예측', color='green', fill=True, alpha=0.4)
plt.xlabel('발전량 (kWh)')
plt.ylabel('밀도')
plt.title('실제 vs 예측 발전량 KDE 곡선 (RandomForest)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()