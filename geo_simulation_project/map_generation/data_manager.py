from shapely import wkt
import geopandas as gpd
import rasterio
from rasterio.features import shapes
import numpy as np
from shapely.geometry import shape

class DataManager:
    # threshold_dem = 0で陸地全体を取得、threshold_dem > 0でthreshold_dem以上の標高のみ取得
    # 海面領域を取得するには threshold_dem = -9999 とする
    def load_dem_polygons_from_geotiff(self, input_file, threshold_dem = 0):
        with rasterio.open(input_file) as src:
            image = src.read(1)
        if threshold_dem == -9999:
            mask = image == -9999
        else:
            mask = image > threshold_dem
        polygons = list(shapes(mask.astype(np.int16), transform=src.transform))
        return [shape(polygon) for polygon, val in polygons if val == 1]
    
    # Shapefileからポリゴンデータを読み込む、人口密集地データなど
    def load_polygons_from_shapefile(self, input_file):
        gdf_did = gpd.read_file(input_file, encoding='UTF-8')  # Shapefile読込
        src_proj, dst_proj = 4612, 2443  # 変換前の座標系(緯度経度座標系), 変換後の座標系(平面直角座標系 I系)
        gdf_did.crs = f'epsg:{src_proj}'  # 変換前座標を指定
        gdf_did = gdf_did.to_crs(epsg=dst_proj)  # 変換後座標に変換
        polygons = gdf_did['geometry']
        return polygons
    
    # ポリゴンデータを保存
    # MATLABでそのまま読み込める形式で保存
    # def save_polygons(self, polygons, polygon_type ,output_file):
    #     with open(output_file, 'w') as f:
    #         f.write("import auxFunctions.*\n\n")
    #         if isinstance(polygons, list):
    #             for polygon in polygons:
    #                 coords = list(polygon.exterior.coords)
    #                 if coords[0] == coords[-1]:
    #                     coords = coords[:-1]
    #                 f.write("poly = polygon(")
    #                 for coord in coords:
    #                     f.write("[" + str(coord[0]/1000) + "; " + str(coord[1]/1000) + "], ")
    #                 f.seek(f.tell() - 2, 0)
    #                 f.write(");\n" + f'map.add_shape_to_region({polygon_type}, poly)\n')
    #         else:
    #             coords = polygons.exterior.coords
    #             if coords[0] == coords[-1]:
    #                 coords = coords[:-1]
    #             f.write("poly = polygon(")
    #             for coord in coords:
    #                 f.write("[" + str(coord[0]/1000) + "; " + str(coord[1]/1000) + "], ")
    #             f.seek(f.tell() - 2, 0)
    #             f.write(");\n" + f'map.add_shape_to_region({polygon_type}, poly)\n')

    def save_polygons(self, polygons ,output_file):
        with open(output_file, 'w') as f:
            if isinstance(polygons, list):
                f.write("vertices = [")
                for i in range(len(polygons)):
                    coords = list(polygons[i].exterior.coords)
                    if coords[0] == coords[-1]:
                        coords = coords[:-1]
                    f.write("polygon(")
                    for coord in coords:
                        f.write("[" + str(coord[0]/1000) + ", " + str(coord[1]/1000) + "], ")
                    f.seek(f.tell() - 2, 0)
                    if i == len(polygons) - 1:
                        f.write(")\n")
                    else:
                        f.write("),\n")
                f.write("]")
            else:
                coords = polygons.exterior.coords
                if coords[0] == coords[-1]:
                    coords = coords[:-1]
                f.write("[")
                for coord in coords:
                    f.write("[" + str(coord[0]/1000) + ", " + str(coord[1]/1000) + "], ")
                f.seek(f.tell() - 2, 0)
                f.write("]\n")
