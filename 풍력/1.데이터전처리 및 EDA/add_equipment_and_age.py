import pandas as pd
from datetime import datetime, date

# CSV 파일 읽기
df = pd.read_csv('한국남동발전_시간대별_풍력_발전실적_조정완료.csv')

print(f"처리 전 데이터 수: {len(df)}")

# 영흥풍력 1호기 데이터 삭제
df = df[~((df['발전구분'] == '영흥풍력') & (df['호기'] == 1))]
print(f"영흥풍력 1호기 삭제 후 데이터 수: {len(df)}")

# 설비용량 정보 딕셔너리
equipment_capacity = {
    ('삼천포풍력', 1): 0.75,
    ('영흥풍력', 2): 3.0,
    ('군위 화산풍력', 1): 1.65,
    ('어음풍력', 1): 4.2
}

# 준공일 정보 딕셔너리 (년.월)
commissioning_dates = {
    ('삼천포풍력', 1): datetime(2018, 5, 1),
    ('영흥풍력', 2): datetime(2013, 5, 1),
    ('군위 화산풍력', 1): datetime(2020, 6, 1),
    ('어음풍력', 1): datetime(2023, 11, 1)
}

# 설비용량 열 추가
df['설비용량(MW)'] = df.apply(lambda row: equipment_capacity.get((row['발전구분'], row['호기']), None), axis=1)

# 일자를 datetime으로 변환
df['일자'] = pd.to_datetime(df['일자'])

# 연식 계산 (소수점 첫째자리 반올림)
def calculate_age(row):
    commissioning_date = commissioning_dates.get((row['발전구분'], row['호기']))
    if commissioning_date:
        # 일수를 년으로 변환 (365.25일 = 1년)
        age_days = (row['일자'] - commissioning_date).days
        age_years = age_days / 365.25
        return round(age_years, 1)
    return None

df['연식(년)'] = df.apply(calculate_age, axis=1)

# 데이터 확인
print("\n발전소별 설비용량과 연식 확인:")
for (plant, unit) in equipment_capacity.keys():
    sample = df[(df['발전구분'] == plant) & (df['호기'] == unit)].iloc[0]
    print(f"{plant} {unit}호기: 설비용량 {sample['설비용량(MW)']}MW, 연식 {sample['연식(년)']}년")

# 결과를 새 파일로 저장
output_file = '한국남동발전_시간대별_풍력_발전실적_최종.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n처리 완료! 결과가 '{output_file}'에 저장되었습니다.")
print(f"최종 데이터 수: {len(df)}")
print(f"컬럼: {list(df.columns)}")

# 샘플 데이터 확인
print(f"\n샘플 데이터 (처음 5행):")
print(df.head())
