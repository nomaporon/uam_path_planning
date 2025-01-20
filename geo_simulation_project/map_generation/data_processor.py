import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union
import cv2
import matplotlib.pyplot as plt
from visualizer import Visualizer

class DataProcessor:
    def __init__(self, min_area=750000, large_area=32000000, divisions=5, min_approx_polygon_area=780000):
        self.min_area = min_area
        self.large_area = large_area
        self.divisions = divisions
        self.min_approx_polygon_area = min_approx_polygon_area

    def process_polygons(self, polygons):
        merged_polygon = unary_union(polygons)
        polygons = list(merged_polygon.geoms)
        polygons = [p for p in polygons if p.area > self.min_area] # データ量削減のため、min_areaより小さいポリゴンは除外
        approx_polygons = []

        for polygon in polygons:
            if polygon.is_empty:
                continue
            coords = np.array(polygon.exterior.coords, dtype=np.float32)
            
            if polygon.area > self.large_area:
                # 近似の精度を上げるため、大きなポリゴンは分割して近似
                approx_polygons.extend(self._divide_and_approximate_polygon(polygon))
            else:
                approx_polygons.append(self._approximate_min_area_rect(coords))
        
        return [p for p in approx_polygons if p.area > self.min_approx_polygon_area]

    def _divide_and_approximate_polygon(self, polygon):
        minx, miny, maxx, maxy = polygon.bounds
        dx ,dy = (maxx - minx) / self.divisions, (maxy - miny) / self.divisions
        approx_polygons = []

        for j in range(self.divisions):
            for k in range(self.divisions):
                b = box(minx + j*dx, miny + k*dy, minx + (j+1)*dx, miny + (k+1)*dy)
                intersection = polygon.intersection(b)
                if intersection.is_empty:
                    continue
                if intersection.geom_type == 'Polygon':
                    intersections = [intersection]
                else:
                    intersections = list(intersection.geoms)

                for inter in intersections:
                    approx_polygons.append(self._approximate_intersection(inter))

        return approx_polygons

    def _approximate_intersection(self, inter):
        if isinstance(inter, Polygon):
            coords = np.array(inter.exterior.coords, dtype=np.float32)
        elif isinstance(inter, MultiPolygon):
            coords = np.array(inter.geoms[0].exterior.coords, dtype=np.float32)
        elif isinstance(inter, LineString):
            coords = np.array(inter.coords, dtype=np.float32)
        else:
            raise TypeError(f"Unexpected geometry type: {type(inter)}")
        
        return self._approximate_min_area_rect(coords)

    def _approximate_min_area_rect(self, coords):
        rect = cv2.minAreaRect(coords)
        box1 = cv2.boxPoints(rect)
        box1 = np.intp(box1)
        return Polygon(box1)
    
    def _approximate_douglas_peucker(self, coords):
        epsilon = 0.004 * cv2.arcLength(coords, True)
        polygon = cv2.approxPolyDP(coords, epsilon, True)
        return Polygon(polygon.squeeze())
    
    def select_polygons(self, polygons=None):
        visualizer = Visualizer()
        if polygons is None:
            plt.figure(), plt.title('Select polygons'),
            plt.xlabel('X'), plt.ylabel('Y')
            plt.grid(True)
            plt.xlim(0, 60000), plt.ylim(-40000, 20000)
        else:
            for polygon in polygons:
                visualizer.add_polygon(polygon, 'blue', 0.35)
        # # # 座標を指定してポリゴンを取得
        get_coords = plt.ginput(-1)
        # # # 指定した座標を用いて一つのポリゴンを作成
        get_polygon = Polygon(get_coords)
        return get_polygon

def test_data_processor():
    from map_generation.data_manager import DataManager
    data_manager = DataManager()
    processor = DataProcessor()

    polygons = data_manager.load_dem_polygons_from_geotiff('../data/raw/nagasaki_geotiff/merge_test.tif', 0)
    get_polygon = processor.select_polygons(polygons)
    # # # ポリゴンの座標をテキストファイルに保存
    with open('../data/raw/selected_polygons.txt', 'w') as f:
        f.write(str(get_polygon) + "\n")

if __name__ == "__main__":
    test_data_processor()
