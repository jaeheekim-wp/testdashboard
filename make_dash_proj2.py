
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
# !pip install statsmodels
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px


# 10~15만 표본 내 가성비 집 찾기 / for 20대 싱글 

# 데이터 불러오기 
house= pd.read_csv("./data/houseprice/houseprice-with-lonlat.csv")

# 사용 변수만 추출하기 
house = house[["Longitude","Latitude",'Gr_Liv_Area','Overall_Cond','Garage_Cars','Year_Built','Neighborhood','Year_Remod_Add','Sale_Price']]
house.columns

# 100000 이상 150000 미만 집값 추출
house1015 = house[(house["Sale_Price"] >= 100000) & (house["Sale_Price"] < 150000) ]
house1015 #1016개 

# 정렬하기 
house1015 = house1015.sort_values("Gr_Liv_Area", ascending = False)
house1015

# -----------------------------------------------------
# 변수별 평가 지표 만들기 =============================
# -----------------------------------------------------

# 1.Gr_Liv_Area 

# Gr_Liv_Area 기준으로 점수 부여 (5점이 가장 넓고, 1점이 가장 좁음)
# 가격 대비 면적 계산 (면적 당 가격을 구하고, 그 역수로 면적 대비 가격 지표를 만듦) 

# 평당 집값 변수 생성
house1015['Price_Per_Area'] = house1015['Sale_Price'] / house1015['Gr_Liv_Area']
house1015

# 평당 집값 기준으로 평가 변수 부여
house1015['Score_GrLivArea'] = np.where(house1015['Price_Per_Area'] <= house1015['Price_Per_Area'].quantile(0.2), 5,
                          np.where(house1015['Price_Per_Area'] <= house1015['Price_Per_Area'].quantile(0.4), 4,
                          np.where(house1015['Price_Per_Area'] <= house1015['Price_Per_Area'].quantile(0.6), 3,
                          np.where(house1015['Price_Per_Area'] <= house1015['Price_Per_Area'].quantile(0.8), 2,
                          1))))
house1015 
# house1015['Score_GrLivArea'] = pd.qcut(house1015['Gr_Liv_Area'], 5, labels=[1, 2, 3, 4, 5])

# 상위/하위 10% 가성비 좋은 집들 보기
GrLivArea_low = house1015['Price_Per_Area'].quantile(0.1)  # 하위 10%의 평당 가격
GrLivArea_high = house1015['Price_Per_Area'].quantile(0.9)  # 상위 10%의 평당 가격

# 가성비 좋은 집들 필터링
best_value_houses = house_2[house_2['Price_Per_Area'] <= GrLivArea_low]
best_value_houses.sort_values('Price_Per_Area').head()


# 시각화

# 01
# 평당 가격 분포 히스토그램
plt.figure(figsize=(10, 10))
plt.rcParams.update({"font.family" : "Malgun Gothic"}) 
sns.histplot(house1015['Price_Per_Area'], bins=30, kde=True, color='skyblue')
plt.title('평당 가격 분포도', fontsize=15)
plt.xlabel('평당 가격', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.show()
plt.clf()

# 02
# 평당 평균 데이터 
score_grouped = house1015.groupby('Score_GrLivArea')['Price_Per_Area'].mean().reset_index()
score_grouped

# 각 점수별 평균 평당 가격 막대그래프 
plt.subplots_adjust(left=0.15, bottom=0.13)  # 여백 값은 필요에 맞게 조정 가능
sns.barplot(data=score_grouped, x='Score_GrLivArea', y='Price_Per_Area', hue ='Score_GrLivArea', palette='deep')
plt.title('각 점수별 평균 평당 가격', fontsize=15)
plt.xlabel('평가 점수', fontsize=12)
plt.ylabel('평균 평당 가격', fontsize=12)
plt.show()
plt.clf()

# 03
# 면적별 가격 분포도 
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Gr_Liv_Area', y='Sale_Price', data=house_2, color='skyblue')
plt.title('Gr_Liv_Area vs Sale_Price')
plt.xlabel('Gr_Liv_Area')
plt.ylabel('Sale_Price')
plt.show()

# 03-1
# 회귀선이 포함된 산점도 그리기
plt.subplots_adjust(left=0.27, bottom=0.17)
sns.regplot(x='Gr_Liv_Area', y='Sale_Price', data=house_2, scatter_kws={'s': 10}, line_kws={"color": "red"})
plt.title('Gr_Liv_Area vs Sale_Price', size =7)
plt.xlabel('Gr_Liv_Area', size =7)
plt.ylabel('Sale_Price', size =7)
plt.show()
plt.clf()

# 04
# 회귀직선 
# fig 활용 
fig = px.scatter(
    house_2,
    x="Gr_Liv_Area",
    y="Sale_Price",
    trendline="ols"
)
# 레이아웃 업데이트 
fig.update_layout(
    title=dict(text="<b>GrLivArea vs SalePrice</b>", font=dict(color="#A1C398")),
    paper_bgcolor="#FEFDED",  # 전체 배경색
    plot_bgcolor="#C6EBC5", # 플롯 영역 배경색
    font=dict(color="#C6EBC5"),
    xaxis=dict(
        title=dict(text="GrLivArea", font=dict(color="#A1C398")),  # x축 제목 글씨 색상
        tickfont=dict(color="#C6EBC5"), # x축 눈금 글씨 색상:
        gridcolor='#FEFDED'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="SalePrice", font=dict(color="#A1C398")), 
        tickfont=dict(color="#C6EBC5"),
        gridcolor='#FEFDED'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="#A1C398"))
)
# 점 크기 및 투명도 설정
fig.update_traces(marker=dict(size=7, opacity = 0.5))
# 시각화 
fig.show()

# -----------------------------------------------------
# 2.Garage_Cars

# 10~15만 내의 평가지표 변수 추출
house1015
house1015.columns

# 시각화

# 01
# 주차가능 차 수에 따른 빈도 막대그래프 

house1015[["Garage_Cars"]]
frequency = house1015.groupby('Garage_Cars').size()
frequency

sns.barplot(frequency)
plt.title('10만~15만 달러 주택의 주차 가능 차 수에 따른 빈도수',size = 8)
plt.xlabel('Garage_Cars', fontsize=7)
plt.show()
plt.clf()

# 02
# 점수 부여 후 분포도 
house1015['Score_GarageCars'] = np.where(house1015['Garage_Cars'] == 0, 1,
                             np.where((house1015['Garage_Cars'] >= 1) & (house1015['Garage_Cars'] <= 2), 3,
                             np.where((house1015['Garage_Cars'] >= 3) & (house1015['Garage_Cars'] <= 5), 5, None)))
house1015[['Score_GarageCars']] 



# -----------------------------------------------------
# 3.Overall_Cond

# 10~15만 내의 평가지표 변수 추출
house1015
house1015.columns

# 숫자형으로 변경해 평가 변수 부여
house1015[["Overall_Cond"]]
house1015["Score_Overall_Cond"] = np.where(house1015["Overall_Cond"] == 'Very_Poor', 1,
                                   np.where(house1015["Overall_Cond"] == "Poor", 1,
                                   np.where(house1015["Overall_Cond"] == "Fair", 2,
                                   np.where(house1015["Overall_Cond"] == "Below_Average", 2,
                                   np.where(house1015["Overall_Cond"] == "Average", 3,
                                   np.where(house1015["Overall_Cond"] == "Above_Average", 3,
                                   np.where(house1015["Overall_Cond"] == "Good", 4,
                                   np.where(house1015["Overall_Cond"] == "Very_Good", 4, 5
house1015[["Score_Overall_Cond"]]


# 시각화

# x,y값 지정 
x = house1015["Score_Overall_Cond"]
y = house1015["Sale_Price"]

# 01
# 점수별 빈도 막대그래프  
sns.countplot(x = x, data = house1015)
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 컨디션 점수의 빈도수", fontsize = 8)
plt.xlabel("점수")
plt.ylabel("빈도")
plt.show()
plt.clf()

# 02
# 점수별 분포 히스토그램 
sns.histplot(x = x, y = y, data = house1015)
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 컨디션 점수")
# plt.xticks(range(1, 6, 1))
plt.xlabel("점수")
plt.ylabel("집값")
plt.show()
plt.clf()

# 03
# 점수에 따른 집값 선그래프 
sns.lineplot(x = x, y = y, data = house1015, errorbar = None, marker = "o")
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 집값별 컨디션 점수")
plt.xticks(range(1, 6, 1))
plt.yticks(range(100000, 150000, 10000))
plt.xlabel("점수")
plt.ylabel("집값")
plt.show()
plt.clf()

# 산점도 필요x(그냥 해봄)
# sns.scatterplot(x = x, y = y, data = house_df)
# plt.rcParams.update({"font.family" : "Malgun Gothic"})    
# plt.xlabel("점수")
# plt.ylabel("집값")
# plt.xticks(range(1, 6, 1))
# plt.show()
# plt.clf()

# -----------------------------------------------------
# 4.Year_Remod_Add

# 10~15만 내의 평가지표 변수 추출
house1015
house1015.columns

# 리모델링 연도에 따른 가성비 집
house1015[["Year_Built"]].max() #2008
house1015[["Year_Built"]].min() #1872

#각 집의 상태를 평가한 점수 부여
def calculate_condition_score(row):
    if row["Year_Remod_Add"] > 2010:
        return 5  # 리모델링이 최근에 이루어진 경우
    elif row["Year_Remod_Add"] > 2000:
        return 4
    elif row["Year_Remod_Add"] > 1990:
        return 3
    elif row["Year_Remod_Add"] > 1980:
        return 2
    else:
        return 1  # 오래된 주택일 경우


# 상태에 따른 평가 변수 추가
house1015["Score_year_remod"] = house1015.apply(calculate_condition_score, axis=1)
house1015[["Score_year_remod"]]

# 가성비 계산 (상태 점수를 가격으로 나눔)
# 높은 값은 상대적으로 저렴한 가격에 상태가 좋은 집
house1015["Value_For_Money"] = house1015["Score_year_remod"] / house1015["Sale_Price"]

# 가성비 높은 순으로 정렬
house1015 = house1015.sort_values("Value_For_Money", ascending=False)
house1015.head(50)
house1015

# 시각화

# 01
# 가성비/판매 가격(Sale_Price) 산점도
sns.scatterplot(
    data=house1015, 
    x="Value_For_Money", 
    y="Sale_Price", 
    hue="Score_year_remod",  # 상태 점수에 따라 색상 구분
    palette="viridis",  # 색상 팔레트
    size="Score_year_remod",  # 점 크기는 Condition_Score에 따라 다르게 설정
    sizes=(20, 200),  # 점 크기 범위 설정
    legend=True
)

plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("가성비 분포", fontsize=5)
plt.xlabel("Score_year_remod", fontsize=7)
plt.ylabel("Value_For_Money", fontsize=7)
# plt.legend(title="Score_year_remod")
plt.grid(True)
plt.show()
plt.clf()


# 02
# 히스토그램 그리기 ( 소수점 범위가 작아서 안그려짐 )
plt.hist(house1015["Value_For_Money"], bins=30, color='skyblue')
plt.xlabel('Value For Money')
plt.ylabel('Frequency')
plt.title('Histogram of Value For Money')

# X축을 로그 스케일로 변환
plt.xscale('log')

# X축 라벨 포맷 설정
plt.xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ['0.00001', '0.0001', '0.001', '0.01', '0.1'])

# X축 범위 설정
plt.ylim(0, 0.0001)
plt.xlim(1e-5, 1e-3)  # 이 범위는 데이터에 맞게 조정

plt.show()
plt.clf()

# -----------------------------------------------------
# 5.Year_Built

# 10~15만 내의 평가지표 변수 추출
house1015
house1015.columns

# 년도 별 점수 부여하기 
 ## 최대/최소값 확인
built_min = house1015['Year_Built'].min() # 1872
built_max = house1015['Year_Built'].max() # 2008
 ## 구간 구하기 
(built_max - built_min) / 5 # 27.2
np.arange(1872, 2009, 27.2)

x_1 = np.arange(1872, 1900)
x_2 = np.arange(1900, 1927)
x_3 = np.arange(1927, 1954)
x_4 = np.arange(1954, 1981)
x_5 = np.arange(1981, 2009)

# 평가 변수 부여하기 
## 방법 1
house1015["Score_Year_Built"] = np.where(house1015["Year_Built"].isin(x_1), 1,
                            np.where(house1015["Year_Built"].isin(x_2), 2,
                            np.where(house1015["Year_Built"].isin(x_3), 3,
                            np.where(house1015["Year_Built"].isin(x_4), 4, 5
                            ))))

house1015[["Score_Year_Built"]]

## 방법 2
# bin_year = [1872, 1900, 1927, 1954, 1981, 2009]
# labels = [1, 2, 3, 4, 5] # bins의 갯수보다 1개 적어야 한다.
# house1015["Score_Year_Built"] = pd.cut(house1015["Year_Built"], bins = bin_year, 
#                                         labels = labels, right = False)


# 시각화

# x,y값 지정 
x1 = house1015["Score_Year_Built"]
y1 = house1015["Sale_Price"]

# 01
# 막대그래프  

sns.countplot(x = x1, data = house1015)
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 건축년도 점수의 빈도수")
plt.xlabel("점수")
plt.ylabel("빈도")
plt.show()
plt.clf()

# 02
# 선그래프

sns.lineplot(x = x1, y = y1, data = house1015, errorbar = None, marker = "o")
plt.rcParams.update({"font.family" : "Malgun Gothic"})    
plt.title("10만 ~ 15만 달러 주택의 집값별 건축년도 점수")
plt.xticks(range(1, 6, 1))
plt.yticks(range(100000, 150000, 10000))
plt.xlabel("점수")
plt.ylabel("집값")
plt.show()
plt.clf()

# -----------------------------------------------------
# KICHEN QUAL - 파일에 없슴! 뭘로 해야하나 
# -----------------------------------------------------
# 종합 평가 변수 

# 종합 평가 점수 계산
house1015['Total_Score'] = (
    house1015['Score_GrLivArea'] + 
    house1015['Score_GarageCars'] + 
    house1015['Score_Overall_Cond'] + 
    house1015['Score_year_remod'] + 
    house1015['Score_Year_Built']) / 5

house1015


# 결과 확인
top_value_houses = house1015[["Longitude", 'Latitude', 'Neighborhood','Sale_Price', 'Score_GrLivArea', 'Score_Overall_Cond',
                                    'Score_GarageCars', 'Score_year_remod','Score_Year_Built', 'Total_Score']]
top_value_houses

# 상위순으로 정렬 
top_value_houses = top_value_houses.sort_values(by='Total_Score', ascending=False)
top_value_houses
# 모든 열을 표시하도록 설정
pd.set_option('display.max_columns', None)
# 기본값으로 되돌리기
pd.reset_option('display.max_columns')













