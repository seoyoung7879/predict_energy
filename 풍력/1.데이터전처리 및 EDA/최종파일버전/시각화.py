
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows용 예시: 'Malgun Gothic')
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 데이터 불러오기
df = pd.read_csv(r'풍력\data\풍력_전처리.csv')


# 1. 발전량(kWh) 분포
plt.figure(figsize=(8,4))
sns.histplot(df['발전량(kWh)'], bins=30, kde=True)
plt.title('발전량(kWh) 분포')
plt.xlabel('발전량(kWh)')
plt.ylabel('빈도')
plt.show()
plt.close()

# 2. 풍속 vs 발전량 산점도
plt.figure(figsize=(8,6))
sns.scatterplot(x='풍속(m/s)', y='발전량(kWh)', data=df, alpha=0.5)
plt.title('풍속과 발전량의 관계')
plt.xlabel('풍속(m/s)')
plt.ylabel('발전량(kWh)')
plt.show()
plt.close()

# 3. 풍향별 평균 발전량
plt.figure(figsize=(10,4))
sns.barplot(x='풍향(16방위)', y='발전량(kWh)', data=df)
plt.title('풍향별 평균 발전량')
plt.xlabel('풍향(16방위)')
plt.ylabel('평균 발전량(kWh)')
plt.show()
plt.close()

# 4. 시간대별 평균 발전량 (시간별 트렌드)
plt.figure(figsize=(10,4))
hourly = df.groupby('시간')['발전량(kWh)'].mean()
hourly.plot(marker='o')
plt.title('시간대별 평균 발전량')
plt.xlabel('시간')
plt.ylabel('평균 발전량(kWh)')
plt.grid(True)
plt.show()
plt.close()

# 5. 설비용량 대비 발전량 산점도
plt.figure(figsize=(8,6))
sns.scatterplot(x='설비용량(MW)', y='발전량(kWh)', data=df, alpha=0.5)
plt.title('설비용량(MW)와 발전량(kWh) 관계')
plt.xlabel('설비용량(MW)')
plt.ylabel('발전량(kWh)')
plt.show()
plt.close()

# 6. 연식(년)별 평균 발전량
plt.figure(figsize=(8,4))
sns.barplot(x='연식(년)', y='발전량(kWh)', data=df)
plt.title('연식(년)별 평균 발전량')
plt.xlabel('연식(년)')
plt.ylabel('평균 발전량(kWh)')
plt.show()
plt.close()

# 7. 주요 변수와 발전량의 상관관계 히트맵
target_cols = ['발전량(kWh)', '설비용량(MW)', '연식(년)', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '적설(cm)', '태양고도']
cols_exist = [col for col in target_cols if col in df.columns]
if len(cols_exist) > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(df[cols_exist].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('주요 변수 상관관계')
    plt.show()
    plt.close()

# 8. 발전구분별 평균 발전량
plt.figure(figsize=(8,4))
sns.barplot(x='발전구분', y='발전량(kWh)', data=df)
plt.title('발전구분별 평균 발전량')
plt.xlabel('발전구분')
plt.ylabel('평균 발전량(kWh)')
plt.show()
plt.close()

# 9. 지점명별 평균 발전량
plt.figure(figsize=(10,4))
sns.barplot(x='지점명', y='발전량(kWh)', data=df)
plt.title('지점명별 평균 발전량')
plt.xlabel('지점명')
plt.ylabel('평균 발전량(kWh)')
plt.xticks(rotation=45)
plt.show()
plt.close()
    