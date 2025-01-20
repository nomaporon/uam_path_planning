import numpy as np
from region_map import RegionMap
from problem import Problem
from solver import Solver
from polygon import polygon
from ball import ball
import matplotlib.pyplot as plt
import utils as ut
import opengen as og
from shapely.geometry import LineString, Point
import geopandas as gpd
import os

class Main:
    def __init__(self):
        self.map = None
        self.problem = None
        self.solver = None

    ### if you change map info, also change map_version to apply the changes
    def setup_map(self):
        self.map = RegionMap()
        # self.map.map_version = 'v1_solver_v2'
        self.map.map_version = 'v1'

        ### No-fly zone
        air_port = ball([38.66652661075855, -9.203164091309498], 9)
        defense_base1 = ball([46.36137256675563, 3.9427562315386298], 2)
        defense_base2 = ball([19.846825121034392, 18.93411773399299], 2)
        defense_base3 = ball([26.037433469490207, 15.46710452712196], 2)
        heli_port = ball([46.87758543585609, -19.138710035318375], 2)
        self.map.add_obstacles(air_port, defense_base1, defense_base2, defense_base3, heli_port)
        # obstacle = ut.get_var_from_file('../../data/processed/no_fly_area.txt', 'vertices')
        # self.map.add_obstacles(*obstacle)

        ### Land
        self.map.new_region('Land', [0.9290, 0.6940, 0.1250])
        land = ut.get_var_from_file('../../data/processed/land_area.txt', 'vertices')
        self.map.add_shapes_to_region('Land', *land)

        ### Population
        self.map.new_region('Population', 'Red')
        populated_area = ut.get_var_from_file('../../data/processed/populated_area.txt', 'vertices')
        self.map.add_shapes_to_region('Population', *populated_area)
        
        ### Historical center
        self.map.new_region('HistCenter', 'Green')
        hist_center = ball([33.874752, -24.981154], 1)
        self.map.add_shape_to_region('HistCenter', hist_center)

    ### if you change some of problem options, you need to update the solver
    ### (at setup_solver_options(), set self.solver.update_solver = True)
    def setup_problem(self):
        N = 80  # Number of waypoints
        opts = {
            'length_smooth': True,
            'penalty_smooth': True,
            'obstacle_smooth': True,
            'maxratio_smooth': False,
        }
        self.problem = Problem(self.map, N, opts)

    ### if you change solver_options, you need to update the solver
    ### (at setup_solver_options(), set self.solver.update_solver = True)
    def setup_solver_options(self):
        build_config = og.config.BuildConfiguration()\
            .with_build_directory("python_build")\
            .with_tcp_interface_config()
        
        optimizer_name = "map_" + self.map.map_version + "_n" + str(self.problem.N)
        meta = og.config.OptimizerMeta()\
            .with_optimizer_name(optimizer_name)
        
        # solver_config = og.config.SolverConfiguration()\
        #     .with_tolerance(1e-4)\
        #     .with_initial_tolerance(1e-3)\
        #     .with_max_duration_micros(5000000)\
        #     .with_max_inner_iterations(1e15)
        solver_config = og.config.SolverConfiguration()\
            .with_tolerance(1e-4)\
            .with_initial_tolerance(1e-3)\
            .with_max_inner_iterations(1000000)
        
        opts = {'build_config': build_config, 'meta': meta, 'solver_config': solver_config}
        self.solver = Solver(self.problem, opts)
        self.solver.optimizer_name = optimizer_name

    def get_solver_result(self, x_init, params):
        result = self.solver.solve(x_init, params)
        return result

    def plot_results(self, result, color):
        self.map.plot(result)
        self.solver.plot(color)
        plt.axis('equal')
        plt.xlim(10, 50)
        plt.ylim(-40, 15)

    def check_options(self, maxratio, maxalpha):
        assert maxratio >= 1
        assert 0 <= maxalpha <= np.pi

    def make_result_line_shp(self, x, file_path, start_point=[35590.685, -27711.422], end_point=[26478.673, 9564.082]):
        points = [start_point] + [(1000*x[i], 1000*x[i+1]) for i in range(0, len(x), 2)] + [end_point]
        line = LineString(points)
        gdf = gpd.GeoDataFrame(geometry=[line], crs='EPSG:2443')
        gdf = gdf.to_crs('EPSG:4612')
        gdf.to_file(file_path)

    def save_points_to_shp(self, x, output_path, start_point=[35590.685, -27711.422], end_point=[26478.673, 9564.082]):
        # Point オブジェクトのリストを作成
        points = [start_point] + [(1000*x[i], 1000*x[i+1]) for i in range(0, len(x), 2)] + [end_point]
        point_geometries = [Point(point) for point in points]
        gdf = gpd.GeoDataFrame(geometry=point_geometries, crs='EPSG:2443')
        gdf = gdf.to_crs('EPSG:4612')
        gdf.to_file(output_path)

    ### params are x_start, x_goal, maxratio, maxalpha, enlargement, weight of each region
    ### you don't rebuild the solver if you change these params
    ### if you change map info, N, some of problem options, you need to rebuild the solver
    ### (at setup_solver_options(), set self.solver.update_solver = True)
    def run(self):
        self.setup_map()
        self.setup_problem()
        self.setup_solver_options()
        
        ### Set start and goal points
        x_start, x_goal =  [35.590685, -27.711422], [26.478673, 9.564082]
        # x_start, x_goal =  [35.590685, -37.711422], [26.478673, 9.564082]
        # x_start, x_goal =  [44.590685, -63.711422], [26.478673, 9.564082]
        self.map.x_start, self.map.x_goal = x_start, x_goal

        maxratio, maxalpha, enlargement = 1.05, np.pi/90, 0 ### N=80
        # maxratio, maxalpha, enlargement = 1.1, np.pi/80, 0 ### N=80
        maxratio, maxalpha, enlargement = 1.04, np.pi/80, 0 ### N=80
        # maxratio, maxalpha, enlargement = 1.05, np.pi/70, 0 ### N=70
        # maxratio, maxalpha, enlargement = 1.05, np.pi/60, 0 ### N=60
        # maxratio, maxalpha, enlargement = 1.1, np.pi/40, 0 ### N=40
        # maxratio, maxalpha, enlargement = 1.1, np.pi/40, 1 
        # maxratio, maxalpha, enlargement = 1.2, np.pi/20, 0 ### N=20
        # maxratio, maxalpha, enlargement = 1.25, np.pi/10, 0 ### N=10
        # maxratio, maxalpha, enlargement = 1.3, np.pi/5, 0 ### N=5
        self.check_options(maxratio, maxalpha)

        weights = [200, 15000, 27000] ### according to the order of regions in setup_map()
        # weights = [100, 7500, 13500]
        # weights = [1600, 120000, 188000]
        # weights = [100, 7000, 12000] ### N=60, 80

        params = (x_start + x_goal + [maxratio, maxalpha, enlargement] + weights)

        ### solver will be updated automatically if map info and points are changed
        ### if you set update_solver to True, solver will be updated
        ### if you change build_config etc., set this to True to update the solver
        self.solver.update_solver = True
        self.solver.update_solver = False

        # displacements = [0] ### 1 initial path
        # displacements = np.arange(-1, 2) / 2 ### 3 initial path, [-0.5, 0, 0.5]
        displacements = np.arange(-2, 3) / 4 ### 5 initial path, [-0.5, -0.25, 0, 0.25, 0.5]

        min_fval, min_fval_index = 0, 0
        min_length, min_length_index = 0, 0
        colors = ['b', 'c', 'k', 'm', 'g', 'y', 'r', 'orange', 'purple', 'brown']
        print('Start simulation: N =', self.problem.N)
        print('Solver', self.solver.optimizer_name)
        print("-------------------------------------")
        for i in range(len(displacements)):
            if i != 0: self.solver.update_solver = False ### use the same solver for the first path
            x_init = self.solver.create_x_init(displacements[i]) ### create initial path
            result = self.get_solver_result(x_init, params) ### solve the problem
            
            # print the result
            print('line', i+1)
            if min_fval == 0 or result['fval'] < min_fval:
                min_fval = result['fval']
                min_fval_index = i
            if min_length == 0 or result['length'] < min_length:
                min_length = result['length']
                min_length_index = i
            info = "time: {} s\nfval: {}\nlength: {} km\nexit_status: {}".format(result['time'], result['fval'], result['length'], result['exit_status'])
            print(info)
            print("-------------------------------------")

            ### make shp file for the result
            self.make_result_line_shp(result['x'], '../../data/simulated/sim_result/line' + str(i+1) + '.shp') ### make result path shp

            # self.make_result_line_shp(x_init, '../../data/simulated/initial_path/initial_path' + str(i+1) + '.shp') ### make initial path shp

            self.save_points_to_shp(result['x'], '../../data/simulated/sim_result/line' + str(i+1) + '_points.shp') ### make result points shp

            # plot the result
            self.plot_results(result, colors[i])

        print('Min fval result: line', min_fval_index+1)
        print('Min path length result: line', min_length_index+1)
        plt.show()

if __name__ == "__main__":
    main = Main()
    main.run()