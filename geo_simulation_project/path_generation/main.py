import numpy as np
from region_map import RegionMap
from problem import Problem
from solver import Solver
import polygon
import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.map = None
        self.problem = None
        self.solver = None

    def setup_map(self):
        self.map = RegionMap()

        # Obstacle
        # 円形の障害物
        # obs_ball1 = polygon.create_ball([2, -2], 1)
        # self.map.add_obstacle(obs_ball1)
        
        # 四角形の障害物
        self.map.new_region('HistCenter', 'Green')
        self.map.new_region('Population', 'Red')
        self.map.new_region('Land', [0.9290, 0.6940, 0.1250])
        obs_rect1 = polygon.create_rectangle(-1, -1, 1, 1)
        # self.map.add_obstacle(obs_rect1)
        self.map.add_shape_to_region('HistCenter', obs_rect1)
        
	    # Population
        # self.map.new_region('Population', 'Red')
        # run('populated_area_vertices.m')

        # #  Historical center
        # self.map.new_region('HistCenter', 'Green')
        # # ball_obstacle = polygon.create_ball([33.874752, -24.981154], 1)  # 中心(0,0)、半径1の円
        # hist_center = polygon.create_ball([1, 1], 1)
        # self.map.add_shape_to_region('HistCenter', hist_center)

        # 領域の設定
        # self.map.new_region('Land', [0.9290, 0.6940, 0.1250])
        # ここで land_vertices.m の内容を Python で実装する必要があります

        # 開始点と目標点の設定
        # self.map.x_start = np.array([35.590685, -27.711422])
        # self.map.x_goal = np.array([26.478673, 9.564082])
        self.map.x_start = np.array([5, -11])
        self.map.x_goal = np.array([-2, 10])

    def setup_problem(self):
        N = 50  # 軌道上の点の数
        opts = {
            'length_smooth': True,
            'penalty_smooth': True,
            'obstacle_smooth': False,
            'maxratio_smooth': False,
            'equality': True,
            'maxratio': 1.1,
            'maxalpha': np.pi/8,
            'enlargement': 0
        }
        self.problem = Problem(self.map, N, opts)
        # self.problem.set_weight('HistCenter', 4)

    def solve(self):
        self.solver = Solver(self.problem)
        x_init = self.solver.create_x_init(0)  # 初期軌道の生成
        result = self.solver.solve(x_init)
        return result

    # def plot_results(self, result):
    #     self.map.plot(result)
    #     self.solver.plot('b')
    #     # self.problem.plot3Dregions(result)

    def plot_results(self, result):
        fig, ax = plt.subplots(figsize=(10, 8))
        self.map.plot(result, ax=ax)
        self.solver.plot('b', ax=ax)
        # self.problem.plot3Dregions(result, ax=ax)
        # ax.set_xlim(-12, 10)
        # ax.set_ylim(-11, 11)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def run(self):
        self.setup_map()
        self.setup_problem()
        result = self.solve()
        # print(result)
        self.plot_results(result)

if __name__ == "__main__":
    main = Main()
    main.run()