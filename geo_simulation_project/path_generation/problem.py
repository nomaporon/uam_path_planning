import numpy as np
from typing import Dict, Callable, Tuple
from region_map import RegionMap
import casadi as cs

class Problem:
    def __init__(self, map: RegionMap, N: int, opts: Dict = None):
        assert isinstance(map, RegionMap)
        self.map = map
        self.N = N
        self.weights: Dict[str, float] = {}
        self.options = {
            'length_smooth': False,
            'penalty_smooth': True,
            'obstacle_smooth': False,
            'maxratio_smooth': False,
        }
        self.params = {
            'maxratio': None,
            'maxalpha': None,
            'enlargement': None    
        }
        if opts:
            self.options.update(opts)
        self.update_weights()

    def update_weights(self):
        for region_name in self.map.region_names():
            if region_name not in self.weights:
                self.weights[region_name] = 1

    def set_weight(self, region_name: str, w: float):
        assert region_name in self.map.regions
        self.weights[region_name] = w

    # cost = distance + penalty from each region
    # cost = Σ_i(|Δz_i| + psi_region1(z_i) + psi_region2(z_i) + ...)
    def get_cost(self, z):
        path_length = self.length_of(z, self.options['length_smooth']) / self.N
        penalty = self.get_total_penalty_function()
        cost = path_length
        for j in range(self.N + 2):
            cost += penalty(z[2*j:2*(j+1)])
        return cost
    
    # get penalty function of each region
    # if you give z, it will return the penalty from all regions at z
    # total_penalty = psi_region1(z) + psi_region2(z) + ...
    def get_total_penalty_function(self) -> Callable:
        def total_penalty(x):
            penalty = 0
            for region_name in self.map.region_names():
                weighted_psi = self.get_penalty_function(region_name)
                penalty += weighted_psi(x)
            return penalty
        return total_penalty
    
    # get_penalty_function(region_name) returns the penalty function of the region
    def get_penalty_function(self, region_name=None):
        if region_name is None:
            Obstacles = self.map.obstacles
            smooth = self.options['obstacle_smooth']
            w = 1
        else:
            self.update_weights()
            Obstacles = self.map.regions[region_name]['shapes']
            smooth = self.options['penalty_smooth']
            w = self.weights[region_name]

        enlargement = self.params['enlargement']

        def penalty(x):
            total = 0
            for obs in Obstacles:
                psi = obs.penalty_function(smooth, enlargement)
                if np.isnan(obs.center).any():
                    total += psi(x)
                else:
                    total += (psi(x) / psi(obs.center))
            return w * total / self.N

        return penalty

    def get_nonlincon(self, z):
        constraints = []
        options_ = self.options
        params_ = self.params
        maxratio_smooth = options_['maxratio_smooth']
        maxratio = params_['maxratio']
        maxalpha = params_['maxalpha']

        N_ = self.N

        nrm = cs.norm_2 if not maxratio_smooth else lambda a: cs.norm_2(a)**2
        if maxratio_smooth:
            maxratio = maxratio**2

        mincos = cs.cos(maxalpha)

        for k in range(N_):
            zk = z[2*(k+1):2*(k+2)] - z[2*k:2*(k+1)]
            zk1 = z[2*(k+2):2*(k+3)] - z[2*(k+1):2*(k+2)]
            constraints.append(cs.fmax(0.0, nrm(zk1) - maxratio * nrm(zk)))
            constraints.append(cs.fmax(0.0, nrm(zk) / maxratio - nrm(zk1)))
            
            cos_theta = cs.dot(zk, zk1) / (nrm(zk) * nrm(zk1))
            constraints.append(cs.fmax(0.0, mincos - cos_theta))

        for obs in self.map.obstacles:
            psi = obs.penalty_function(options_['obstacle_smooth'])
            for j in range(self.N + 2):
                constraints.append(psi(z[2*j:2*(j+1)]))

        return cs.vertcat(*constraints)

    # def feasibility_of(self, x: np.ndarray) -> Tuple[float, float]:
    #     if x.shape[0] == 2:
    #         x = np.tile(x, (self.N, 1)).flatten()
    #         divby = self.N
    #     else:
    #         divby = 1

    #     c = self.get_nonlincon()
    #     ineq, eq = c(x)
    #     eq = np.max(np.abs(eq)) / divby
    #     ineq = np.max(np.maximum(0, ineq)) / divby
    #     return float(ineq), float(eq)
    
    # x = [z1[0], z1[1], z2[0], z2[1], ..., zN-1[0], zN-1[1]]  2*N × 1
    def length_of(self, x, smooth=False):
        if smooth:
            nrm = lambda y: cs.sumsqr(y)
        else:
            nrm = cs.norm_2

        x_reshaped = cs.reshape(x, (-1 , 1))
        x_start = cs.reshape(self.map.x_start, (-1, 1))
        x_goal = cs.reshape(self.map.x_goal, (-1, 1))

        y = cs.vertcat(x_start, x_reshaped, x_goal)
        out = 0
        for k in range(self.N+1):
            yk = y[2*k:2*k+2]
            yk1 = y[2*k+2:2*k+4]
            out += nrm(yk1-yk)
        return out

    # def plot3D(self, *args, **kwargs):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D

    #     axislim = self.map.get_axislim()
    #     xs = np.linspace(axislim[0], axislim[1], 50)
    #     ys = np.linspace(axislim[2], axislim[3], 50)
    #     X, Y = np.meshgrid(xs, ys)

    #     f = self.get_total_penalty_function()
    #     Z = np.zeros_like(X)
    #     for i in range(Z.shape[0]):
    #         for j in range(Z.shape[1]):
    #             Z[i,j] = f(np.array([X[i,j], Y[i,j]]))

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     surf = ax.plot_surface(X, Y, Z, *args, **kwargs)
    #     plt.colorbar(surf)

    #     xs = self.map.x_start
    #     xg = self.map.x_goal
    #     z = max(f(xs), f(xg)) + 1
    #     ax.plot([xs[0], xs[0]], [xs[1], xs[1]], [0, z], 'k-o', markersize=10, linewidth=2)
    #     ax.plot([xg[0], xg[0]], [xg[1], xg[1]], [0, z], 'k-*', markersize=10, linewidth=2)

    #     plt.show()

    # def plot3Dregions(self, *args, **kwargs):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D

    #     axislim = self.map.get_axislim()
    #     xs = np.linspace(axislim[0], axislim[1], 50)
    #     ys = np.linspace(axislim[2], axislim[3], 50)
    #     X, Y = np.meshgrid(xs, ys)

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     for name in self.map.region_names():
    #         color = self.map.regions[name]['color']
    #         f = self.get_penalty_function(name)
    #         Z = np.zeros_like(X)
    #         for i in range(Z.shape[0]):
    #             for j in range(Z.shape[1]):
    #                 Z[i,j] = f(np.array([X[i,j], Y[i,j]]))

    #         surf = ax.plot_surface(X, Y, Z, color=color, *args, **kwargs)

    #     # Plot the base plane
    #     ax.plot_surface(X, Y, np.zeros_like(X), color='white')

    #     # Plot start and end points
    #     f = self.get_total_penalty_function()
    #     xs = self.map.x_start
    #     xg = self.map.x_goal
    #     z = max(f(xs), f(xg)) + 1
    #     ax.plot([xs[0], xs[0]], [xs[1], xs[1]], [0, z], 'k-o', markersize=10, linewidth=2)
    #     ax.plot([xg[0], xg[0]], [xg[1], xg[1]], [0, z], 'k-*', markersize=10, linewidth=2)

    #     plt.show()

if __name__ == "__main__":
    from polygon import polygon
    from ball import ball
    import matplotlib.pyplot as plt
    map = RegionMap()
    N = 10
    map.x_start = [35.590685, -27.711422]
    map.x_goal = [26.478673, 9.564082]
    z0, zN = map.x_start, map.x_goal
    z = (np.linspace(z0, zN, N + 2)[1:-1]).flatten()
    # z = cs.SX.sym('z', 2*N)
    problem = Problem(map, N)

    land = [polygon([16.088709677419356, 11.006493506493506],[12.21774193548387, -7.8246753246753284],[28.245967741935484, -27.629870129870138],[33.20564516129032, -16.83441558441559],[28.48790322580645, 1.9967532467532438]),
            polygon([25.04032258064516, -24.464285714285722],[33.931451612903224, -38.26298701298702],[48.14516129032258, -22.43506493506494],[34.596774193548384, -12.207792207792211]),
            polygon([46.08870967741936, -18.214285714285722],[46.99596774193549, 18.474025974025963],[36.41129032258064, 4.756493506493506],[39.979838709677416, -9.853896103896107]),
            polygon([14.818548387096774, 19.36688311688311],[29.334677419354833, 19.52922077922078],[27.520161290322577, 14.41558441558441],[15.483870967741934, 10.519480519480515]),
            polygon([25.826612903225804, 18.717532467532468],[29.879032258064516, 4.918831168831166],[40.28225806451613, 5.649350649350643],[46.391129032258064, 18.555194805194798])
            ]
    map.new_region('Land', [0.9290, 0.6940, 0.1250])
    map.add_shapes_to_region('Land', *land)

    populated_area = [polygon([28.836, -32.708], [29.607, -36.124], [32.53, -35.464], [31.759, -32.048]),
        polygon([29.185, -27.687], [29.35, -29.043], [32.001, -28.719], [31.835, -27.362]),
        polygon([30.45, -24.494], [32.097, -24.809], [32.293, -23.781], [30.646, -23.466]),
        polygon([30.179, -22.099], [31.872, -22.221], [32.013, -20.255], [30.32, -20.134]),
        polygon([31.826, -33.217], [32.92, -33.773], [34.082, -31.487], [32.989, -30.931]),
        polygon([32.221, -29.916], [33.536, -29.916], [33.536, -28.183], [32.221, -28.183]),
        polygon([32.871, -26.783], [32.871, -28.183], [33.536, -28.183], [33.536, -26.783]),
        polygon([32.313, -25.511], [32.313, -26.28], [33.536, -26.28], [33.536, -25.511]),
        polygon([31.875, -25.511], [33.536, -25.511], [33.536, -22.838], [31.875, -22.838]),
        polygon([31.875, -20.349], [31.875, -22.838], [33.536, -22.838], [33.536, -20.349]),
        polygon([33.536, -30.856], [33.536, -33.302], [35.197, -33.302], [35.197, -30.856]),
        polygon([33.536, -28.183], [33.536, -30.856], [35.197, -30.856], [35.197, -28.183]),
        polygon([33.536, -25.511], [33.536, -28.183], [35.197, -28.183], [35.197, -25.511]),
        polygon([33.536, -22.838], [33.536, -25.511], [35.197, -25.511], [35.197, -22.838]),
        polygon([33.536, -22.106], [33.536, -22.838], [34.618, -22.838], [34.618, -22.106]),
        polygon([35.197, -28.183], [35.197, -30.856], [36.858, -30.856], [36.858, -28.183]),
        polygon([35.197, -28.183], [36.858, -28.183], [36.858, -25.511], [35.197, -25.511]),
        polygon([34.658, -24.157], [36.628, -24.963], [37.229, -23.493], [35.259, -22.687]),
        polygon([36.858, -28.183], [36.858, -30.578], [37.654, -30.578], [37.654, -28.183]),
        polygon([36.327, -27.292], [37.673, -28.391], [38.569, -27.293], [37.222, -26.195]),
        polygon([24.856, -18.992], [25.294, -20.24], [27.054, -19.623], [26.616, -18.375]),
        polygon([31.691, -19.726], [35.922, -20.728], [36.67, -17.565], [32.439, -16.563]),
        polygon([26.334, 16.705], [27.688, 13.096], [30.145, 14.018], [28.791, 17.627]),
        polygon([14.398, 21.212], [20.922, 13.157], [27.408, 18.41], [20.884, 26.465]),
        polygon([40.009, -26.859], [42.78, -28.265], [44.413, -25.048], [41.642, -23.642]),
        polygon([34.034, -20.664], [35.549, -21.925], [36.207, -21.134], [34.693, -19.873]),
        polygon([43.479, -19.521], [44.456, -20.493], [46.884, -18.051], [45.907, -17.079]),
        polygon([37.098, -8.131], [41.568, -14.026], [46.327, -10.417], [41.858, -4.522]),
        polygon([32.392, 7.207], [33.374, 5.991], [35.54, 7.741], [34.558, 8.957])
    ]
    map.new_region('Population', 'Red')
    map.add_shapes_to_region('Population', *populated_area)

    map.new_region('HistCenter', 'Green')
    hist_center = ball([33.874752, -24.981154], 1)
    map.add_shape_to_region('HistCenter', hist_center)

    problem.set_weight('Land', 4)
    problem.set_weight('Population', 13)
    problem.set_weight('HistCenter', 45)

    map.plot()

    print(problem.get_cost(z))
    print(problem.length_of(z))
    print(problem.get_nonlincon(z))
    plt.axis('equal')
    plt.xlim(10, 50)
    plt.ylim(-40, 15)
    plt.show()

    # z = cs.SX.sym('z', 2*N)
    # z = np.zeros(2*N)
    # t = np.linspace(map.x_start[0], map.x_goal[0], N+2)[1:-1]
    # z[0::2] = t
    # t = np.linspace(map.x_start[1], map.x_goal[1], N+2)[1:-1]
    # z[1::2] = t
    # print(z)
    # print(problem.length_of(z))