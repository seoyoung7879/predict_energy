import pandas as pd
import numpy as np

# CSV 파일 읽기 (인코딩 문제 해결)
try:
    df = pd.read_csv('한국남동발전_시간대별_풍력_발전실적.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('한국남동발전_시간대별_풍력_발전실적.csv', encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv('한국남동발전_시간대별_풍력_발전실적.csv', encoding='utf-8-sig')

print("원본 데이터 컬럼:")
print(df.columns.tolist())
print(f"원본 데이터 행 수: {len(df)}")

# 시간별 발전량 컬럼들 찾기 (1시~24시)
hour_columns = []
for i in range(1, 25):
    for col in df.columns:
        if f'{i}시' in col and '발전량' in col:
            hour_columns.append(col)
            break

print(f"시간별 발전량 컬럼: {hour_columns}")

# 기본 정보 컬럼들 - 존재하는 모든 컬럼 확인
print("실제 CSV 파일의 모든 컬럼:")
for i, col in enumerate(df.columns):
    print(f"{i+1}: '{col}'")

# 기본 정보 컬럼들 찾기 (정확한 컬럼명 매칭)
base_columns = []
for col in df.columns:
    col_stripped = col.strip()
    if any(keyword in col_stripped for keyword in ['발전구분', '구분']):
        base_columns.append(col)
    elif any(keyword in col_stripped for keyword in ['호기', '호']):
        base_columns.append(col)
    elif any(keyword in col_stripped for keyword in ['일자', '날짜', 'date']):
        base_columns.append(col)

print(f"찾은 기본 컬럼: {base_columns}")

# 만약 기본 컬럼을 못 찾았다면 처음 3개 컬럼을 사용
if len(base_columns) < 3:
    print("기본 컬럼을 정확히 찾지 못했습니다. 처음 3개 컬럼을 사용합니다.")
    non_hour_cols = [col for col in df.columns if not any(f'{i}시' in col for i in range(1, 25)) 
                     and not any(keyword in col for keyword in ['총량', '평균', '최대', '최소'])]
    base_columns = non_hour_cols[:3]
    print(f"사용할 기본 컬럼: {base_columns}")

# 새로운 데이터프레임 생성을 위한 리스트
new_data = []

# 각 행에 대해 시간별로 데이터 분해
for idx, row in df.iterrows():
    base_info = {col: row[col] for col in base_columns}
    
    for hour, hour_col in enumerate(hour_columns, 1):
        if hour_col in df.columns:
            new_row = base_info.copy()
            new_row['시간'] = hour
            new_row['발전량(MWh)'] = row[hour_col]
            new_data.append(new_row)

# 새로운 데이터프레임 생성
new_df = pd.DataFrame(new_data)

# 일자 컬럼에서 시간 부분 제거 (년월일만 남기기)
for col in new_df.columns:
    if any(keyword in col for keyword in ['일자', '날짜', 'date']):
        # datetime 형식으로 변환 후 날짜만 추출
        try:
            new_df[col] = pd.to_datetime(new_df[col]).dt.date
        except:
            # 만약 변환이 안 되면 문자열에서 날짜 부분만 추출
            new_df[col] = new_df[col].astype(str).str.split(' ').str[0]
        break

# 컬럼 순서 정리
column_order = base_columns + ['시간', '발전량(MWh)']
new_df = new_df[column_order]

print(f"변환된 데이터 행 수: {len(new_df)}")
print("변환된 데이터 컬럼:")
print(new_df.columns.tolist())

# 변환된 데이터의 처음 몇 행 확인
print("\n변환된 데이터 미리보기:")
print(new_df.head(30))

# 새로운 CSV 파일로 저장
new_df.to_csv('한국남동발전_시간대별_풍력_발전실적_재구성.csv', index=False, encoding='utf-8-sig')
print("\n파일이 '한국남동발전_시간대별_풍력_발전실적_재구성.csv'로 저장되었습니다.")

# 각 발전소별 데이터 확인
print("\n발전소별 데이터 개수:")
print(new_df['발전구분'].value_counts())
