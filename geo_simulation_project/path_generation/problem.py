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
            'equality': True,
            'maxratio': 1.1,
            'maxalpha': np.pi/8,
            'enlargement': 0
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

    def get_cost(self):
        def cost(u):
            path_length = self.length_of(u, self.options['length_smooth']) / self.N
            penalty = self.get_total_penalty_function()
            total_cost = path_length
            for j in range(self.N):
                total_cost += penalty(u[2*j:2*(j+1)])
            return cs.sum1(total_cost)
        return cost
    
    def get_total_penalty_function(self) -> Callable:
        def total_penalty(x):
            penalty = 0
            for region_name in self.map.region_names():
                weighted_psi = self.get_penalty_function(region_name)
                penalty += weighted_psi(x)
            return penalty
        return total_penalty

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

        enlargement = self.options['enlargement']

        def penalty(x):
            x_reshaped = cs.reshape(x, (-1, 2))
            total = cs.SX.zeros(1)
            for obs in Obstacles:
                psi = obs.penalty_function(smooth, enlargement)
                total += cs.sum1(psi(x_reshaped))
            return w * total / self.N

        return penalty

    def get_nonlincon(self):
        self.check_options()
        options_ = self.options
        maxratio_smooth = options_['maxratio_smooth']
        equality = options_['equality']
        maxratio = options_['maxratio']
        maxalpha = options_['maxalpha']

        N_ = self.N
        psi = self.get_penalty_function()

        def obs_penalty(x):
            return cs.sum1(cs.vertcat(*[psi(x[2*j:2*(j+1)]) for j in range(N_)]))

        x0 = cs.reshape(self.map.x_start, (1, 2))
        xf = cs.reshape(self.map.x_goal, (1, 2))

        nrm = cs.norm_2 if not maxratio_smooth else lambda a: cs.norm_2(a)**2
        if maxratio_smooth:
            maxratio = maxratio**2

        mincos = cs.cos(maxalpha)

        def c(u):
            x_start = cs.reshape(self.map.x_start, (1, 2))
            x_goal = cs.reshape(self.map.x_goal, (1, 2))
            u_reshaped = cs.reshape(u, (self.N, 2))
            y = cs.vertcat(x_start, u_reshaped, x_goal)
            
            ineq = []
            eq = []
            
            for k in range(self.N):
                yk = y[k+1, :] - y[k, :]
                yk1 = y[k+2, :] - y[k+1, :]
                if self.options['maxratio'] > 1:
                    ineq.append(cs.norm_2(yk) - self.options['maxratio'] * cs.norm_2(yk1))
                    ineq.append(cs.norm_2(yk1) - self.options['maxratio'] * cs.norm_2(yk))
                else:
                    eq.append(cs.norm_2(yk) - cs.norm_2(yk1))
                
                if self.options['maxalpha'] < cs.pi - 1e-4:
                    ineq.append(cs.cos(self.options['maxalpha']) * cs.norm_2(yk) * cs.norm_2(yk1) - cs.dot(yk, yk1))
    
            obs_pen = self.get_penalty_function()(u)
            if self.options['equality']:
                eq.append(obs_pen)
            else:
                ineq.append(obs_pen)
    
            return cs.vertcat(*ineq), cs.vertcat(*eq)
    
        return c

    def check_options(self):
        assert self.options['maxratio'] >= 1
        assert 0 <= self.options['maxalpha'] <= np.pi

    def feasibility_of(self, x: np.ndarray) -> Tuple[float, float]:
        if x.shape[0] == 2:
            x = np.tile(x, (self.N, 1)).flatten()
            divby = self.N
        else:
            divby = 1

        c = self.get_nonlincon()
        ineq, eq = c(x)
        eq = np.max(np.abs(eq)) / divby
        ineq = np.max(np.maximum(0, ineq)) / divby
        return float(ineq), float(eq)

    def length_of(self, x, smooth=False):
        nrm = cs.norm_2 if not smooth else lambda x: cs.norm_2(x)**2
        x_reshaped = cs.reshape(x, (self.N, 2))

        # Ensure all elements are 2D vectors
        x_start = cs.reshape(self.map.x_start, (1, 2))
        x_goal = cs.reshape(self.map.x_goal, (1, 2))

        y = cs.vertcat(x_start, x_reshaped, x_goal)
        return cs.sum1(cs.vertcat(*[nrm(y[k+1, :] - y[k, :]) for k in range(self.N+1)]))

    def plot3D(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        axislim = self.map.get_axislim()
        xs = np.linspace(axislim[0], axislim[1], 50)
        ys = np.linspace(axislim[2], axislim[3], 50)
        X, Y = np.meshgrid(xs, ys)

        f = self.get_total_penalty_function()
        Z = np.zeros_like(X)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i,j] = f(np.array([X[i,j], Y[i,j]]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, *args, **kwargs)
        plt.colorbar(surf)

        xs = self.map.x_start
        xg = self.map.x_goal
        z = max(f(xs), f(xg)) + 1
        ax.plot([xs[0], xs[0]], [xs[1], xs[1]], [0, z], 'k-o', markersize=10, linewidth=2)
        ax.plot([xg[0], xg[0]], [xg[1], xg[1]], [0, z], 'k-*', markersize=10, linewidth=2)

        plt.show()

    def plot3Dregions(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        axislim = self.map.get_axislim()
        xs = np.linspace(axislim[0], axislim[1], 50)
        ys = np.linspace(axislim[2], axislim[3], 50)
        X, Y = np.meshgrid(xs, ys)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for name in self.map.region_names():
            color = self.map.regions[name]['color']
            f = self.get_penalty_function(name)
            Z = np.zeros_like(X)
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[i,j] = f(np.array([X[i,j], Y[i,j]]))

            surf = ax.plot_surface(X, Y, Z, color=color, *args, **kwargs)

        # Plot the base plane
        ax.plot_surface(X, Y, np.zeros_like(X), color='white')

        # Plot start and end points
        f = self.get_total_penalty_function()
        xs = self.map.x_start
        xg = self.map.x_goal
        z = max(f(xs), f(xg)) + 1
        ax.plot([xs[0], xs[0]], [xs[1], xs[1]], [0, z], 'k-o', markersize=10, linewidth=2)
        ax.plot([xg[0], xg[0]], [xg[1], xg[1]], [0, z], 'k-*', markersize=10, linewidth=2)

        plt.show()