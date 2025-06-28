import pandas as pd
import numpy as np
# 파일 경로
file_path = r'풍력\data\풍력_전처리.csv'

# 데이터 불러오기
df = pd.read_csv(file_path)

# 1. 데이터 크기
print("Shape:", df.shape)

# 2. 컬럼명 및 데이터 타입
print("\nColumns and dtypes:\n", df.dtypes)

# 3. 결측치 개수
print("\nMissing values:\n", df.isnull().sum())

# 4. 기초 통계
print("\nDescribe:\n", df.describe(include='all'))

# 5. 샘플 데이터
print("\nHead:\n", df.head())


# 발전량(kWh)을 MW로 변환 후 소수점 둘째자리에서 버림
df['발전량(MW)'] = np.floor(df['발전량(kWh)'] / 1000 * 100) / 100

# 설비용량(kW) 컬럼 추가 (설비용량(MW) * 1000)
df['설비용량(kW)'] = df['설비용량(MW)'] * 1000


# 이상치 판별: 발전량(MW) > 설비용량(MW)
outliers_power = df[df['발전량(MW)'] > df['설비용량(MW)']]
print(f"\n[발전량 이상치] 개수: {len(outliers_power)}")
print(outliers_power[['발전량(kWh)', '발전량(MW)', '설비용량(MW)']].head())



# 이상치 행을 새로운 파일로 저장 (설비용량(kW) 컬럼 포함)
df_clean = df[df['발전량(MW)'] < df['설비용량(MW)']].copy()
df_clean['설비용량(kW)'] = df_clean['설비용량(MW)'] * 1000
df_clean.to_csv(r'풍력\\data\\풍력_전처리_이상치제거.csv', index=False)
print(f"\n이상치 제거 후 데이터 shape: {df_clean.shape}")

# 이상치가 잘 제거됐는지 확인을 위해 파일을 다시 불러옴
df_check = pd.read_csv(r'풍력\\data\\풍력_전처리_이상치제거.csv')
check_outliers = df_check[df_check['발전량(MW)'] > df_check['설비용량(MW)']]
print(f"\n[이상치 제거 후 남은 발전량 이상치 개수]: {len(check_outliers)}")
if len(check_outliers) > 0:
    print(check_outliers[['발전량(kWh)', '발전량(MW)', '설비용량(MW)']].head())
else:
    print("이상치가 모두 제거되었습니다.")

# 풍속(m/s) 이상치 (예: 음수 또는 50m/s 초과)
if '풍속(m/s)' in df.columns:
    outliers_ws = df[(df['풍속(m/s)'] < 0) | (df['풍속(m/s)'] > 35)]
    print(f"\n[풍속 이상치] 개수: {len(outliers_ws)}")
    print(outliers_ws[['풍속(m/s)']].head())

# 풍향(16방위) 이상치 (예: 0~15 이외 값)
if '풍향(16방위)' in df.columns:
    outliers_wd = df[(df['풍향(16방위)'] < 0) | (df['풍향(16방위)'] > 360)]
    print(f"\n[풍향 이상치] 개수: {len(outliers_wd)}")
    print(outliers_wd[['풍향(16방위)']].head())

# 기온(°C) 이상치 (예: -40도 미만, 60도 초과)
if '기온(°C)' in df.columns:
    outliers_temp = df[(df['기온(°C)'] < -30) | (df['기온(°C)'] > 50)]
    print(f"\n[기온 이상치] 개수: {len(outliers_temp)}")
    print(outliers_temp[['기온(°C)']].head())

# 습도(%) 이상치 (예: 0~100 이외 값)
if '습도(%)' in df.columns:
    outliers_hum = df[(df['습도(%)'] < 0) | (df['습도(%)'] > 100)]
    print(f"\n[습도 이상치] 개수: {len(outliers_hum)}")
    print(outliers_hum[['습도(%)']].head())
