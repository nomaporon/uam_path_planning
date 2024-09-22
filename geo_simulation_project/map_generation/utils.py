import random
from matplotlib import patches
from shapely.geometry import Point
import geopandas as gpd

def generate_random_colors(n):
    return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(n)]

def add_legend(ax, colors):
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', alpha=0.5, label=f'Polygon {i+1}') 
                       for i, color in enumerate(colors)]
    ax.legend(handles=legend_elements, loc='upper right')

def make_points_shp():
    # スタートとゴールの座標を指定してShapefileを作成
    start_point_coords, end_point_coords = [32.749507,129.879793], [33.08592134,129.78364085]
    # 座標からPointオブジェクトを作成
    start_point, end_point = Point(start_point_coords[::-1]), Point(end_point_coords[::-1]) # 経度、緯度の順にする
    gdf = gpd.GeoDataFrame(geometry=[start_point, end_point], crs='EPSG:6668')
    gdf = gdf.to_crs('EPSG:3857')
    gdf['geometry'] = gdf['geometry'].buffer(300)  # bufferの単位はメートル
    gdf = gdf.to_crs('EPSG:4612')
    gdf.to_file("../data/processed/generated_points/start_and_end_points.shp")

def make_peace_park_shp():
    # 平和公園の座標を指定してShapefileを作成
    # 中心座標
    center_coords = [33874.75203033371, -24981.154191253154]
    # 半径1キロの円を作成（座標系がメートル単位であると仮定）
    circle = Point(center_coords).buffer(1000)
    # GeoDataFrameを作成（初期の座標系はEPSG:2443）
    gdf = gpd.GeoDataFrame(geometry=[circle], crs='EPSG:2443')
    # 座標系をEPSG:4612に変換
    gdf = gdf.to_crs('EPSG:4612')
    gdf.to_file("../data/processed/generated_points/peace_park.shp")

# if __name__ == '__main__':
    # make_points_shp()
    # make_peace_park_shp()
