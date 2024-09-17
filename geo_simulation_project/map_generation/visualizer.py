import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from map_generation.utils import generate_random_colors
from geopandas import GeoSeries

class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()

    def set_plot_limits(self, x_min=0, x_max=60000, y_min=-40000, y_max=20000):
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('X', fontsize=10)
        plt.ylabel('Y', fontsize=10)

    def add_polygon(self, polygon, color, alpha):
        if isinstance(polygon, GeoSeries):
            for geom in polygon:
                if geom.geom_type == 'Polygon':
                    xy = np.array(geom.exterior.coords.xy).T
                    poly = patches.Polygon(xy, fc=color, ec='black', alpha=alpha)
                    self.ax.add_patch(poly)
        elif hasattr(polygon, 'exterior') and polygon.exterior is not None:
            xy = np.array(polygon.exterior.coords.xy).T
            poly = patches.Polygon(xy, fc=color, ec='black', alpha=alpha)
            self.ax.add_patch(poly)

    def plot_polygons(self, polygons, color='blue', alpha=0.35):
        for polygon in polygons:
            self.add_polygon(polygon, color, alpha)
        plt.show()

    # # # sets_of_polygons: ポリゴンのリストのリスト [[Polygon, Polygon, ...], [Polygon, ...], ...]
    def plot_multiple_sets_of_polygons(self, sets_of_polygons, colors=['blue'], alpha=0.35):
        if colors is None:
            colors = generate_random_colors(len(sets_of_polygons))
        elif len(colors) < len(sets_of_polygons):
            colors += generate_random_colors(len(sets_of_polygons) - len(colors))

        for polygons, color in zip(sets_of_polygons, colors):
            if isinstance(polygons, list):
                for polygon in polygons:
                    self.add_polygon(polygon, color, alpha)
            else:
                self.add_polygon(polygons, color, alpha)
        plt.show()

def test_visualizer():
    from map_generation.data_manager import DataManager
    from map_generation.data_processor import DataProcessor

    data_manager = DataManager()
    processor = DataProcessor()
    visualizer = Visualizer()

    polygons = data_manager.load_polygons_from_shapefile('../data/raw/populated_area/populated_area.shp')
    processed_polygons = processor.process_polygons(polygons)

    # # # plot_polygonsメソッドをテスト
    # visualizer.plot_polygons(test_polygons, color='green', alpha=0.5)

    test_polygons = []
    test_polygons.append(processed_polygons)
    test_polygons.append(polygons)
    # visualizer.plot_multiple_sets_of_polygons(test_polygons)
    visualizer.plot_multiple_sets_of_polygons(test_polygons, colors=['blue', 'red'])

if __name__ == "__main__":
    test_visualizer()