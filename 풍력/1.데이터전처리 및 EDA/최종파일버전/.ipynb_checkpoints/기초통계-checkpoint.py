import pandas as pd

# 파일 경로
file_path = r'풍력\data\df_merged_wind_final.csv'

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