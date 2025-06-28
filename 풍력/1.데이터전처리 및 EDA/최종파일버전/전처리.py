import pandas as pd

# 파일 경로
file_path = r'풍력\data\df_merged_wind_final.csv'

# 데이터 불러오기
df = pd.read_csv(file_path)

# 1. 선형 보간 적용할 변수 목록
interpolate_cols = ['풍속(m/s)', '풍향(16방위)', '기온(°C)', '습도(%)', '하늘상태']
for col in interpolate_cols:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
# 2. 방위각 컬럼 제거
if '방위각' in df.columns:
    df = df.drop(columns=['방위각'])

# 전처리 결과 확인
print(df.head())
print(df.isnull().sum())
# 전처리된 데이터 저장
df.to_csv(r'풍력\data\풍력_전처리.csv', index=False)
print("\n전처리된 파일이 '풍력_전처리.csv'로 저장되었습니다.")
