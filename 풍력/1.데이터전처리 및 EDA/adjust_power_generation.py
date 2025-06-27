import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('한국남동발전_시간대별_풍력_발전실적_재구성.csv')

# 열 이름 확인 및 공백 제거
print("기존 열 이름:", df.columns.tolist())
df.columns = df.columns.str.strip()
print("정리된 열 이름:", df.columns.tolist())

# 데이터 타입 확인
print("\n데이터 타입:")
print(df.dtypes)

# 각 발전소별 조건에 따라 발전량 조정
# 영흥풍력 2호기 -> 8로 나누기
mask_younghung = (df['발전구분'] == '영흥풍력') & (df['호기'] == 2)
df.loc[mask_younghung, '발전량(kWh)'] = df.loc[mask_younghung, '발전량(kWh)'] / 8

# 군위 화산풍력 1호기 -> 7로 나누기
mask_gunwi = (df['발전구분'] == '군위 화산풍력') & (df['호기'] == 1)
df.loc[mask_gunwi, '발전량(kWh)'] = df.loc[mask_gunwi, '발전량(kWh)'] / 7

# 어음풍력 1호기 -> 5로 나누기
mask_eoeum = (df['발전구분'] == '어음풍력') & (df['호기'] == 1)
df.loc[mask_eoeum, '발전량(kWh)'] = df.loc[mask_eoeum, '발전량(kWh)'] / 5

# 변경된 데이터 확인
print("\n영흥풍력 2호기 조정 후 샘플:")
print(df[mask_younghung].head())

print("\n군위 화산풍력 1호기 조정 후 샘플:")
print(df[mask_gunwi].head())

print("\n어음풍력 1호기 조정 후 샘플:")
print(df[mask_eoeum].head())

# 수정된 데이터를 새 파일로 저장
output_filename = '한국남동발전_시간대별_풍력_발전실적_조정완료.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n조정된 데이터가 '{output_filename}'에 저장되었습니다.")

# 변경 사항 요약
print("\n=== 변경 사항 요약 ===")
print(f"영흥풍력 2호기: {mask_younghung.sum()}개 행의 발전량을 8로 나누었습니다.")
print(f"군위 화산풍력 1호기: {mask_gunwi.sum()}개 행의 발전량을 7로 나누었습니다.")
print(f"어음풍력 1호기: {mask_eoeum.sum()}개 행의 발전량을 5로 나누었습니다.")
