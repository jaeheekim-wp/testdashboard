
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
# house_train=pd.read_csv("./data/houseprice/train.csv")
# house_test=pd.read_csv("./data/houseprice/test.csv")
# sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

house_lonlat = pd.read_csv("./data/houseprice/houseprice-with-lonlat.csv")
house_lonlat.columns

# 필요한 변수만 뽑기 
lonlat = house_lonlat[["Longitude","Latitude",'Gr_Liv_Area','Neighborhood','Sale_Price']]
lonlat

# 집값 오름차순 
lonlat = lonlat.sort_values("Sale_Price", ascending = False)

# # 동네별 평균 집값- 내림차순 
# lonlat_n = lonlat.groupby("Neighborhood", as_index=False) \
#                .agg(price_mean=("Sale_Price", "mean"),
#                     lon_mean=("Longitude", "mean"),
#                     lat_mean=("Latitude", "mean"))

# 상하위지역 구별하기 (평균으로 )

# 상위 3
# lonlat.query('Neighborhood == "Northridge")
# lonlat.query('Neighborhood == "Stone_Brook"')
# lonlat.query('Neighborhood == "Northridge_Heights"')
# 필터링 조건을 적용하여 top_3 변수에 재할당
top_3 = lonlat[lonlat["Neighborhood"].isin(["Northridge", "Stone_Brook", "Northridge_Heights"])]
top_3

# 하위 3
# lonlat.query('Neighborhood == "Meadow_Village"')
# lonlat.query('Neighborhood == "Iowa_DOT_and_Rail_Road"')
# lonlat.query('Neighborhood == "Briardale"')
# 필터링 조건을 적용하여 low_3 변수에 재할당
low_3 = lonlat[lonlat["Neighborhood"].isin(["Meadow_Village", "Iowa_DOT_and_Rail_Road", "Briardale"])]
low_3

# ----------------------------------------------------------
# 비싼동네 싼 동네 통일 - 주영 
house_sorted = house.sort_values(by='Sale_Price', ascending=False)
house_sorted

hs = house_sorted["Sale_Price"]
hs.head(20)
hs.tail(20)

# 비싼동네 vs 싼동네
house_sorted["Neighborhood"].head(50) #> Northridge, Northridge_Heights, Stone_Brook
house_sorted["Neighborhood"].tail(60) #> Iowa_DOT_and_Rail_Road, Old_Town

--------------------------------
house_sorted = house.sort_values(by='Sale_Price', ascending=False)
house_sorted

hs = house_sorted["Sale_Price"]
hs.head(20)
hs.tail(20)

# 비싼동네 vs 싼동네
house_sorted["Neighborhood"].head(50) #> Northridge, Northridge_Heights, Stone_Brook
house_sorted["Neighborhood"].tail(60) #> Iowa_DOT_and_Rail_Road, Old_Town

# ---------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# 시각화2(모든 변수)
a = lonlat.groupby('Neighborhood')['Gr_Liv_Area'].mean().sort_values()
a

# 바 컬러 바꾸기 
bar_colors = np.where(a["Gr_Liv_Area"]>= 1800,"red",np.where(a["Gr_Liv_Area"] <= 1200 ,"blue","grey"))

# plt.figure(figsize=(14, 8))
sns.barplot(data = a, x = "Neighborhood", y = "Gr_Liv_Area", \
palette=bar_colors)

plt.title('Average of Gr_Liv_Area by Neighborhood Type', fontsize=10)
plt.xlabel('Neighborhood', fontsize=7)
plt.ylabel('Average of Gr_Liv_Area', fontsize=7)

# X축 레이블 각도 및 글씨 크기 조정
plt.xticks(rotation=40, ha='right', fontsize=3)
plt.show()
plt.clf()

------------------------------------------------
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# !pip install statsmodels
import statsmodels.api as sm

# -----------------------
# 상위 회귀직선 
fig = px.scatter(
    top_3,
    x="Gr_Liv_Area",
    y="Sale_Price",
    color="Neighborhood",
    trendline="ols"
)

fig.show()

# 레이아웃 업데이트 
fig.update_layout(
    title=dict(text="<b>< TOP_3 > GrLivArea vs SalePrice</b>", font=dict(color="#A1C398")),
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

# ------------------

# 하위 회귀직선 
fig2 = px.scatter(
    low_3,
    x="Gr_Liv_Area",
    y="Sale_Price",
    color="Neighborhood",
    trendline="ols"
)

# fig.show()

# 레이아웃 업데이트 
fig2.update_layout(
    title=dict(text="<b>< LOW_3 > GrLivArea vs SalePrice</b>", font=dict(color="#F1D3CE")),
    paper_bgcolor="#F6EACB",  # 전체 배경색
    plot_bgcolor="#F1D3CE", # 플롯 영역 배경색
    font=dict(color="#C6EBC5"),
    xaxis=dict(
        title=dict(text="GrLivArea", font=dict(color="#F1D3CE")),  # x축 제목 글씨 색상
        tickfont=dict(color="#FEECAD5"), # x축 눈금 글씨 색상:
        gridcolor='#FEFDED'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="SalePrice", font=dict(color="#F1D3CE")), 
        tickfont=dict(color="#FEECAD5"),
        gridcolor='#FEFDED'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="#F1D3CE"))
)

# 점 크기 및 투명도 설정
fig2.update_traces(marker=dict(size=7, opacity = 0.5))

# 시각화 
fig2.show()


import folium
lonlat.groupby("Neighborhood")

# 흰 도화지 맵 가져오기 
map_sig = folium.Map(location = [42.034, -93.642],
                     zoom_start = 12,
                     tiles = 'cartodbpositron')
                     
# 지도 내보내기                    
# map_sig.save('map_houseprice.html')

# 집값 전체 좌표 찍기 (for 반복문 활용)
# 마커 클러스터 추가
# marker_cluster = MarkerCluster().add_to(map_sig)

for i in range(len(upper_30)):
    folium.CircleMarker(location=[upper_30.iloc[i, 1], upper_30.iloc[i, 0]],
                        radius=5,
                        color='#F6E96B',
                        fill=True,
                        fill_color='#F6E96B').add_to(map_sig)
                        
for i in range(len(lower_30)):
    folium.CircleMarker(location=[lower_30.iloc[i, 1], lower_30.iloc[i, 0]],
                        radius=5,
                        color='#A2CA71',
                        fill=True,
                        fill_color='#A2CA71').add_to(map_sig)                       

# 특정 구역 표시 

# circle
#location1=lonlat.iloc[0:5,1].mean()
#location2=lonlat.iloc[0:5,0].mean()
folium.Circle(
    location=[42.054, -93.623],
    radius=500,  # 반경 (미터 단위)
    color='#FFDA76',
    fill=True,
    fill_color='#FFDA76'
).add_to(map_sig)

# Polygon
# location3=lonlat.iloc[2925:-1,1].mean()
# location4=lonlat.iloc[2925:-1,0].mean()
# folium.Polygon(
#     locations=[41.988,-93.603],[41.998, -93.603], [41.988 , -93.613]],
#     color='#FF8C9E',
#     fill=True,
#     fill_color='#FF8C9E'
# ).add_to(map_sig)

# Marker with custom icon
folium.Marker(
    location=[42.0289, -93.6104],
    popup='outlier',
    icon=folium.Icon(icon='info-sign', color='lightred')
).add_to(map_sig)

## {'gray', 'beige', 'darkpurple', 'lightgreen', 'darkgreen', 'purple', 
# 'darkred', 'cadetblue', 'darkblue', 'black', 'lightred', 'red', 'orange', 
# 'pink', 'white', 'blue', 'lightblue', 'green', 'lightgray'}.

# LatLngPopup 개별 위경도 팝업 
folium.LatLngPopup().add_to(map_sig)

# 지도 저장
map_sig.save('map_fordash.html')















