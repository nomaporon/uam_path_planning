import numpy as np
from region_map import RegionMap
from problem import Problem
from solver import Solver
from polygon import polygon
from ball import ball
import matplotlib.pyplot as plt
import utils as ut

class Main:
    def __init__(self):
        self.map = None
        self.problem = None
        self.solver = None

    def setup_map(self):
        self.map = RegionMap()

        # Obstacle
        # obs_ball1 = polygon.create_ball([2, -2], 1)
        # self.map.add_obstacle(obs_ball1)

        # Land
        self.map.new_region('Land', [0.9290, 0.6940, 0.1250])
        land = ut.get_var_from_file('../../data/processed/land_area.txt', 'vertices')
        self.map.add_shapes_to_region('Land', *land)

        # Population
        self.map.new_region('Population', 'Red')
        populated_area = ut.get_var_from_file('../../data/processed/populated_area.txt', 'vertices')
        self.map.add_shapes_to_region('Population', *populated_area)
        
        # Historical center
        self.map.new_region('HistCenter', 'Green')
        hist_center = ball([33.874752, -24.981154], 1)
        self.map.add_shape_to_region('HistCenter', hist_center)

        # 開始点と目標点の設定
        self.map.x_start = np.array([35.590685, -27.711422])
        self.map.x_goal = np.array([26.478673, 9.564082])

    def setup_problem(self):
        N = 20  # 軌道上の点の数
        opts = {
            'length_smooth': True,
            'penalty_smooth': True,
            'obstacle_smooth': False,
            'maxratio_smooth': False,
            'maxratio': 1.1,
            'maxalpha': np.pi/6,
            'enlargement': 0
        }
        self.problem = Problem(self.map, N, opts)
        self.problem.set_weight('Land', 4)
        self.problem.set_weight('Population', 13)
        self.problem.set_weight('HistCenter', 45)

    def solve(self, x_init):
        result = self.solver.solve(x_init)
        return result

    def plot_results(self, result, color):
        self.map.plot(result)
        self.solver.plot(color)
        plt.axis('equal')
        plt.xlim(10, 50)
        plt.ylim(-40, 15)
        # plt.show()

    def run(self):
        self.setup_map()
        self.setup_problem()
        self.solver = Solver(self.problem)
        displacements = np.arange(-1, 2) / 2 # [-0.5, 0, 0.5]
        # displacements = np.arange(-2, 3) / 4 # [-0.5, -0.25, 0, 0.25, 0.5]
        # displacements = [0]
        colors = ['b', 'r', 'k', 'm', 'g', 'y']
        for i in range(len(displacements)):
            x_init = self.solver.create_x_init(displacements[i])
            result = self.solve(x_init)
            print(result)
            self.plot_results(result, colors[i])
        plt.show()

if __name__ == "__main__":
    main = Main()
    main.run()