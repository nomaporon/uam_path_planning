import numpy as np
from region_map import RegionMap
from problem import Problem
from solver import Solver
from polygon import polygon
from ball import ball
import matplotlib.pyplot as plt
import utils as ut
import opengen as og

class Main:
    def __init__(self):
        self.map = None
        self.problem = None
        self.solver = None

    def setup_map(self):
        self.map = RegionMap()
        ### if you change map info, also change map_version to apply the changes
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

    def setup_problem(self):
        N = 30  # Number of waypoints
        ### if you change some of 'length_smooth', 'penalty_smooth', 'obstacle_smooth', 'maxratio_smooth',
        ### you need to update the solver(at setup_solver_options(), set self.solver.update_solver = True)
        ### you don't need to update the solver if you change maxratio, maxalpha, enlargement, weight of each region
        opts = {
            'length_smooth': True,
            'penalty_smooth': True,
            'obstacle_smooth': True,
            'maxratio_smooth': False,
        }
        self.problem = Problem(self.map, N, opts)

    def setup_solver_options(self):
        build_config = og.config.BuildConfiguration()\
            .with_build_directory("python_build")\
            .with_tcp_interface_config()
        
        optimizer_name = "map_" + self.map.map_version + "_n" + str(self.problem.N)
        meta = og.config.OptimizerMeta()\
            .with_optimizer_name(optimizer_name)
        
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

    ### params are x_start, x_goal, maxratio, maxalpha, enlargement, weight of each region
    ### you don't rebuild the solver if you change these params
    ### if you change map info, N, some of problem options, you need to rebuild the solver(at setup_solver_options(), set self.solver.update_solver = True)
    def run(self):
        self.setup_map()
        self.setup_problem()
        self.setup_solver_options()
        
        ### Set start and goal points
        x_start, x_goal =  [35.590685, -27.711422], [26.478673, 9.564082]
        self.map.x_start, self.map.x_goal = x_start, x_goal
        # maxratio, maxalpha, enlargement = 1.1, np.pi/60, 0 # N=50
        maxratio, maxalpha, enlargement = 1.1, np.pi/20, 0 # N=40
        # maxratio, maxalpha, enlargement = 1.1, np.pi/8, 0
        self.check_options(maxratio, maxalpha)
        weights = [0.1, 26, 45] # according to the order of regions in setup_map()

        params = (x_start + x_goal + [maxratio, maxalpha, enlargement] + weights)

        ### solver will be updated automatically if map info and points are changed
        ### if you set update_solver to True, solver will be updated
        ### if you change build_config etc., set this to True to update the solver
        self.solver.update_solver = True
        self.solver.update_solver = False

        # displacements = [0]
        displacements = np.arange(-1, 2) / 2 # [-0.5, 0, 0.5]
        # displacements = np.arange(-2, 3) / 4 # [-0.5, -0.25, 0, 0.25, 0.5]
        colors = ['b', 'r', 'k', 'm', 'g', 'y']
        for i in range(len(displacements)):
            if i != 0: self.solver.update_solver = False 
            x_init = self.solver.create_x_init(displacements[i])
            result = self.get_solver_result(x_init, params)
            print(result)
            self.plot_results(result, colors[i])
        plt.show()

if __name__ == "__main__":
    main = Main()
    main.run()