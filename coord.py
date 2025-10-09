import pandas as pd
import matplotlib.pyplot as plt
import math
import mplcursors
import matplotlib

# CHINESE
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC']
matplotlib.rcParams['axes.unicode_minus'] = False

excel_file = "5数据资料 5-天津新能源智能网联汽车产业集群数据集.xlsx"
sheet_name = "集群企业名单"
lon_col, lat_col = "经度", "纬度"
name_col = "企业名称"


df = pd.read_excel(excel_file, sheet_name=sheet_name)
df = df.dropna(subset=[lon_col, lat_col])

#平均值取值点
lat_center = df[lat_col].mean()
lon_center = df[lon_col].mean()
print("中心点经纬度：", lat_center, lon_center)

#换算
R = 6371  # km
lat0 = math.radians(lat_center)
lon0 = math.radians(lon_center)

def project(lat, lon):
    lat = math.radians(lat)
    lon = math.radians(lon)
    x = R * (lon - lon0) * math.cos(lat0)
    y = R * (lat - lat0)
    return x, y

df["x_km"], df["y_km"] = zip(*df.apply(lambda row: project(row[lat_col], row[lon_col]), axis=1))

#绘图
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df["x_km"], df["y_km"], s=3, c="blue", alpha=0.6)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Eastings (km)")
ax.set_ylabel("Northings (km)")
ax.set_title("企业位置分布图（悬停）")

#显示
cursor = mplcursors.cursor(scatter, hover=True)
@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    sel.annotation.set(text=df.iloc[idx][name_col])
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

plt.show()
