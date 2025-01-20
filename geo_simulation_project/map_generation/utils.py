import random
from matplotlib import patches
from shapely.geometry import Point, Polygon
import geopandas as gpd
import numpy as np
import os

def generate_random_colors(n):
    return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(n)]

def add_legend(ax, colors):
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', alpha=0.5, label=f'Polygon {i+1}') 
                       for i, color in enumerate(colors)]
    ax.legend(handles=legend_elements, loc='upper right')

def create_star(center, size):
    """中心座標とサイズを指定して星型のPolygonを作成"""
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 2  # 角度を調整して星型が傾かないようにする
    points = []
    for i, angle in enumerate(angles):
        r = size if i % 2 == 0 else size / 2
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def make_start_point_shp():
    start_point_coords = [32.749507, 129.879793]
    start_point = Point(start_point_coords[::-1])
    
    # GeoDataFrameを作成（初期の座標系はEPSG:4326）
    gdf = gpd.GeoDataFrame(geometry=[start_point], crs='EPSG:4326')
    
    # 座標系をEPSG:3857に変換
    gdf = gdf.to_crs('EPSG:3857')
    
    # bufferの単位はメートル
    gdf['geometry'] = gdf['geometry'].buffer(500)
    
    # 座標系をEPSG:4612に変換
    gdf = gdf.to_crs('EPSG:4612')
    
    # 出力ディレクトリの存在確認
    output_dir = "../../data/processed/generated_points/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Shapefileを保存
    gdf.to_file(os.path.join(output_dir, "start_point.shp"))

def make_end_point_shp():
    # スタートとゴールの座標を指定
    end_point_coords = [33.08592134, 129.78364085]
    
    # 星型のオブジェクトを作成
    end_star = create_star(end_point_coords[::-1], 0.005)  # 経度、緯度の順にする
    
    # GeoDataFrameを作成（初期の座標系はEPSG:4326）
    gdf = gpd.GeoDataFrame(geometry=[end_star], crs='EPSG:4326')
    
    # 座標系をEPSG:3857に変換
    gdf = gdf.to_crs('EPSG:3857')
    
    # bufferの単位はメートル
    gdf['geometry'] = gdf['geometry'].buffer(100)
    
    # 座標系をEPSG:4612に変換
    gdf = gdf.to_crs('EPSG:4612')
    
    # 出力ディレクトリの存在確認
    output_dir = "../../data/processed/generated_points/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Shapefileを保存
    gdf.to_file(os.path.join(output_dir, "end_point.shp"))

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

def make_area_shp(polygons):
    gdf = gpd.GeoDataFrame(geometry=polygons,crs='EPSG:2443')
    gdf = gdf.to_crs('EPSG:4612')
    gdf.to_file("../../data/processed/populated_area/populated_area.shp")

def make_no_fly_zone_shp():
    # 平面直角座標系 (EPSG:2443)
    crs_src = 'EPSG:2443'
    crs_dest = 'EPSG:4612'  # 地理座標系 (緯度経度)

    # 座標定義
    locations = {
        "air_port": ([38666.52661075855, -9203.164091309498], 9000),
        "defense_base1": ([46361.37256675563, 3942.7562315386298], 2000),
        "defense_base2": ([19846.825121034392, 18934.11773399299], 2000),
        "defense_base3": ([26037.433469490207, 15467.10452712196], 2000),
        "heli_port": ([46877.58543585609, -19138.710035318375], 2000)
    }

    # バッファ生成
    geometries = [
        Point(coord).buffer(radius)
        for coord, radius in locations.values()
    ]

    # GeoDataFrame作成と座標変換
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs_src)
    gdf = gdf.to_crs(crs_dest)

    # ファイル出力
    gdf.to_file("../../data/processed/no_fly_zone/no_fly_zone.shp")


if __name__ == '__main__':
    # make_start_point_shp()
    # make_end_point_shp()
    # make_peace_park_shp()
    make_no_fly_zone_shp()
