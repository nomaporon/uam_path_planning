from shapely.geometry import Point
import geopandas as gpd
from pyproj import Proj, transform

# 緯度経度座標系 (EPSG:4326)
latlong_proj = Proj(init='epsg:4612')

# 平面座標系 (例として UTM Zone 52N, EPSG:32652)
utm_proj = Proj(init='epsg:2443')

# 緯度経度座標
# center_coords = [33.0345672222222, 129.9963325] # 自衛隊演習場
# center_coords = [33.17055748,129.7128032] # 海上自衛隊
# center_coords = [33.1391631111111,129.779081] # 海上自衛隊射撃基地
# center_coords = [32.91632475,129.913402888889] # 長崎空港
center_coords = [32.8264097,130.0006849] # 諫早駅らへんのヘリポート？

# 緯度経度座標を平面座標に変換
x, y = transform(latlong_proj, utm_proj, center_coords[1], center_coords[0])

print("平面座標系の座標:", x/1000, y/1000)



# center_coords = [33.0345672222222,129.9963325]

# # 半径1キロの円を作成（座標系がメートル単位であると仮定）
# circle = Point(center_coords[::-1])

# # GeoDataFrameを作成（初期の座標系はEPSG:2443）
# gdf = gpd.GeoDataFrame(geometry=[circle], crs='EPSG:6668')

# # 座標系をEPSG:4612に変換
# gdf = gdf.to_crs('EPSG:4612')

# # 変換後の座標を出力
# transformed_coords = gdf.geometry.iloc[0].coords[0]
# print("Transformed coordinates:", transformed_coords)