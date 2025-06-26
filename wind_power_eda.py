# 풍력 발전량 데이터 EDA (탐색적 데이터 분석)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
print("=" * 50)
print("풍력 발전량 데이터 EDA")
print("=" * 50)

df = pd.read_csv('한국남동발전_시간대별_풍력_발전실적_재구성.csv')

# 1. 데이터 기본 정보
print("\n1. 데이터 기본 정보")
print("-" * 30)
print(f"데이터 형태: {df.shape}")
print(f"행 수: {df.shape[0]:,}")
print(f"열 수: {df.shape[1]}")
print("\n컬럼 정보:")
print(df.info())

# 2. 데이터 타입 및 결측값 확인
print("\n2. 데이터 타입 및 결측값")
print("-" * 30)
print("데이터 타입:")
print(df.dtypes)
print("\n결측값 개수:")
print(df.isnull().sum())
print(f"\n전체 결측값 비율: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")

# 3. 컬럼명 정리 (공백 제거)
df.columns = df.columns.str.strip()
print("\n정리된 컬럼명:")
print(df.columns.tolist())

# 4. 발전구분별 기본 통계
print("\n3. 발전구분별 기본 정보")
print("-" * 30)
print("발전구분별 데이터 개수:")
print(df['발전구분'].value_counts())
print("\n호기별 데이터 개수:")
print(df['호기'].value_counts())

# 5. 발전량 기본 통계
print("\n4. 발전량 기본 통계")
print("-" * 30)
print(df['발전량(MWh)'].describe())

# 6. 일자 범위 확인
print("\n5. 데이터 기간")
print("-" * 30)
df['일자'] = pd.to_datetime(df['일자'])
print(f"시작일: {df['일자'].min()}")
print(f"종료일: {df['일자'].max()}")
print(f"전체 기간: {(df['일자'].max() - df['일자'].min()).days + 1}일")

# 7. 시간대별 기본 통계
print("\n6. 시간대별 기본 정보")
print("-" * 30)
print("시간대별 데이터 개수:")
print(df['시간'].value_counts().sort_index())

# 8. 발전소별 발전량 통계
print("\n7. 발전소별 발전량 통계")
print("-" * 30)
plant_stats = df.groupby('발전구분')['발전량(MWh)'].agg([
    'count', 'mean', 'std', 'min', 'max', 'sum'
]).round(2)
print(plant_stats)

# 9. 시각화
print("\n8. 데이터 시각화 생성 중...")
print("-" * 30)

# 그래프 크기 설정
fig = plt.figure(figsize=(20, 15))

# 1) 발전소별 총 발전량
plt.subplot(3, 3, 1)
plant_total = df.groupby('발전구분')['발전량(MWh)'].sum().sort_values(ascending=False)
plt.bar(range(len(plant_total)), plant_total.values)
plt.title('발전소별 총 발전량', fontsize=12)
plt.xlabel('발전소')
plt.ylabel('총 발전량 (MWh)')
plt.xticks(range(len(plant_total)), plant_total.index, rotation=45, ha='right')
for i, v in enumerate(plant_total.values):
    plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=8)

# 2) 시간대별 평균 발전량
plt.subplot(3, 3, 2)
hourly_avg = df.groupby('시간')['발전량(MWh)'].mean()
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=4)
plt.title('시간대별 평균 발전량', fontsize=12)
plt.xlabel('시간')
plt.ylabel('평균 발전량 (MWh)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 25))

# 3) 발전량 분포 (히스토그램)
plt.subplot(3, 3, 3)
non_zero_generation = df[df['발전량(MWh)'] > 0]['발전량(MWh)']
plt.hist(non_zero_generation, bins=50, alpha=0.7, edgecolor='black')
plt.title('발전량 분포 (0 제외)', fontsize=12)
plt.xlabel('발전량 (MWh)')
plt.ylabel('빈도')

# 4) 월별 발전량 분포
plt.subplot(3, 3, 4)
df['월'] = df['일자'].dt.month
monthly_generation = df.groupby('월')['발전량(MWh)'].sum()
plt.bar(monthly_generation.index, monthly_generation.values, color='lightblue', edgecolor='navy')
plt.title('월별 총 발전량', fontsize=12)
plt.xlabel('월')
plt.ylabel('총 발전량 (MWh)')
plt.xticks(monthly_generation.index)
for i, v in zip(monthly_generation.index, monthly_generation.values):
    plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=8)


# 5) 발전소별 박스플롯
plt.subplot(3, 3, 5)
# 0이 아닌 발전량만 사용
non_zero_df = df[df['발전량(MWh)'] > 0]
sns.boxplot(data=non_zero_df, x='발전구분', y='발전량(MWh)')
plt.title('발전소별 발전량 분포 (0 제외)', fontsize=12)
plt.xticks(rotation=45, ha='right')

# 7) 0 발전량 비율
plt.subplot(3, 3, 7)
zero_ratio = df.groupby('발전구분').apply(lambda x: (x['발전량(MWh)'] == 0).sum() / len(x) * 100)
plt.bar(range(len(zero_ratio)), zero_ratio.values)
plt.title('발전소별 무발전 시간 비율', fontsize=12)
plt.xlabel('발전소')
plt.ylabel('무발전 비율 (%)')
plt.xticks(range(len(zero_ratio)), zero_ratio.index, rotation=45, ha='right')
for i, v in enumerate(zero_ratio.values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('풍력발전_EDA_결과.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 추가 분석 결과 출력
print("\n9. 주요 발견사항")
print("-" * 30)

# 발전소별 가동률 (0이 아닌 시간의 비율)
print(f"\n발전소별 가동률 (발전량 > 0인 시간 비율):")
for plant in df['발전구분'].unique():
    plant_data = df[df['발전구분'] == plant]
    operation_rate = (plant_data['발전량(MWh)'] > 0).sum() / len(plant_data) * 100
    print(f"  - {plant}: {operation_rate:.1f}%")

# 시간대별 발전 특성
peak_hour = hourly_avg.idxmax()
min_hour = hourly_avg.idxmin()
print(f"\n시간대별 발전 특성:")
print(f"  - 최대 발전 시간대: {peak_hour}시 ({hourly_avg[peak_hour]:.2f} MWh)")
print(f"  - 최소 발전 시간대: {min_hour}시 ({hourly_avg[min_hour]:.2f} MWh)")

# 전체 통계 요약
total_generation = df['발전량(MWh)'].sum()
avg_generation = df['발전량(MWh)'].mean()
print(f"\n전체 발전량 통계:")
print(f"  - 총 발전량: {total_generation:,.2f} MWh")
print(f"  - 평균 발전량: {avg_generation:.2f} MWh")
print(f"  - 무발전 시간 비율: {(df['발전량(MWh)'] == 0).sum() / len(df) * 100:.1f}%")

print("\n=" * 50)
print("EDA 완료! 그래프가 저장되었습니다.")
print("=" * 50)
