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
df = pd.read_csv('풍력\data\풍력_전처리_이상치제거.csv')
df = df[df['발전구분'] != '영흥풍력']
target_col = '발전량(kWh)'

Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]

# 사용할 피처 및 타겟
feature_cols = [
    '설비용량(kW)', '연식(년)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '적설(cm)', '태양고도',
     '시간', '하늘상태'
]
target_col = '발전량(kWh)'

# 결측치 제거
df = df.dropna(subset=feature_cols + [target_col])

# 범주형 변수 지정
cat_features = [ '시간', '풍향(16방위)', '하늘상태']


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