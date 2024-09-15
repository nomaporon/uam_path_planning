from geo_simulation_project.data_manager import DataManager
from geo_simulation_project.data_processor import DataProcessor
from geo_simulation_project.visualizer import Visualizer
from shapely import wkt

class Main:
    def __init__(self):
        self.data_manager = DataManager()
        self.processor = DataProcessor()
        self.visualizer = Visualizer()

    def main(self):
        # self.process_population() # 人口密集地データの処理
        # self.process_land() # 標高データの処理
        # self.process_select_polygons() # 選択したポリゴンデータの処理
        self.visualize_map() # マップの可視化

    # # # 人口密集地データの処理
    def process_population(self):
        # # # 元データの読み込み
        polygons = self.data_manager.load_polygons_from_shapefile('data/raw/populated_area/populated_area.shp')
        # # # ポリゴンデータの近似処理
        processed_polygons = self.processor.process_polygons(polygons)
        # # # 処理後のポリゴンデータをシミュレーション用に保存
        self.data_manager.save_polygons(processed_polygons, 'Population', 'data/processed/populated_area.txt')
        # # # プロットの表示
        # self.visualizer.plot_polygons(processed_polygons)
        plot_polygons = [polygons, processed_polygons]
        self.visualizer.set_plot_limits(20000, 50000, -40000, -10000)
        self.visualizer.plot_multiple_sets_of_polygons(plot_polygons, colors=['blue', 'red'])

    # # # 標高データの処理
    def process_land(self):
        # # # 元データの読み込み
        # # # 第二引数は標高の閾値、0で陸地全体を取得、0より大きい値でその値以上の標高のみ取得、-9999で海面領域を取得
        polygons = self.data_manager.load_dem_polygons_from_geotiff('data/raw/nagasaki_geotiff/merge_test.tif', 0)
        # # # ポリゴンデータの近似処理
        processed_polygons = self.processor.process_polygons(polygons)
        # # # 処理後のポリゴンデータをシミュレーション用に保存
        self.data_manager.save_polygons(processed_polygons, 'Land', 'data/processed/land_area.txt')
        # # # プロットの表示
        self.visualizer.plot_polygons(polygons)

    # # # 選択したポリゴンデータを処理
    # # # 複数エリアを選択したい場合は、テキストファイルの中身を消さずに再度実行する。リセットする場合はテキストファイルの中身を消す
    # # # 選択を終了するには、Macの場合はoptionキーを押しながらクリック、Windowsの場合はctrlキーを押しながらクリック
    def process_select_polygons(self):
        # # # 陸地データのプロットをもとにポリゴンデータを選択
        land_polygons = self.data_manager.load_dem_polygons_from_geotiff('data/raw/nagasaki_geotiff/merge_test.tif', 0)
        get_polygons = self.processor.select_polygons(land_polygons)
        # # # 選択したポリゴンデータを元データとして保存
        with open('data/raw/selected_polygons.txt', 'a') as f:
            f.write(str(get_polygons) + "\n")
        # # # 選択したポリゴンデータをシミュレーション用に保存
        with open('data/raw/selected_polygons.txt', 'r') as f:
            get_polygons = [wkt.loads(line.strip()) for line in f]
        self.data_manager.save_polygons(get_polygons, 'Land', 'data/processed/selected_area.txt')

    # # # マップの可視化
    def visualize_map(self):
        # # # 人口密集地データの取得
        populated_polygons = self.data_manager.load_polygons_from_shapefile('data/raw/populated_area/populated_area.shp')
        processed_populated_polygons = self.processor.process_polygons(populated_polygons)
        # # # 選択したポリゴンデータの取得
        with open('data/raw/selected_polygons.txt', 'r') as f:
            get_polygons = [wkt.loads(line.strip()) for line in f]

        self.visualizer.plot_multiple_sets_of_polygons([processed_populated_polygons, get_polygons], colors=['blue', 'red'])

if __name__ == "__main__":
    main = Main()
    main.main()
