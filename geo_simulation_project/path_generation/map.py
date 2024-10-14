import numpy as np
from typing import List, Tuple, Optional
from quadratic_obstacle import QuadraticObstacle
import matplotlib.pyplot as plt

class Map:
    def __init__(self, *obstacles):
        self.obstacles: List[QuadraticObstacle] = []
        self.x_goal: np.ndarray = np.zeros(2)
        self.x_start: np.ndarray = np.zeros(2)
        self.add(*obstacles)

    def add(self, *obstacles):
        """Add obstacles to the map."""
        for obstacle in obstacles:
            assert isinstance(obstacle, QuadraticObstacle), "Obstacle must be a QuadraticObstacle object"
            self.obstacles.append(obstacle)

    def intersection(self, x0: np.ndarray, direction: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Return the closest point along a direction.

        Args:
            x0: Starting position.
            direction: Direction vector.

        Returns:
            Tuple of collision point (or None if no collision) and distance from x0.
        """
        dist = float('inf')
        p = None
        for obs in self.obstacles:
            pj = obs.intersection(x0, direction)
            if pj is not None:
                distj = np.linalg.norm(pj - x0)
                if distj < dist:
                    p = pj
                    dist = distj
        return p, dist

    def collides(self, x: np.ndarray) -> bool:
        """Check if a point belongs to any obstacle."""
        return any(obs.contains(x) for obs in self.obstacles)

    def get_axislim(self) -> Tuple[float, float, float, float]:
        """Get the axis limits for plotting."""
        xmin, xmax = self.x_start[0], self.x_start[0]
        ymin, ymax = self.x_start[1], self.x_start[1]

        xmin, xmax = min(xmin, self.x_goal[0]), max(xmax, self.x_goal[0])
        ymin, ymax = min(ymin, self.x_goal[1]), max(ymax, self.x_goal[1])

        for obs in self.obstacles:
            xlim = obs.xlim
            ylim = obs.ylim
            xmin, xmax = min(xmin, xlim[0]), max(xmax, xlim[1])
            ymin, ymax = min(ymin, ylim[0]), max(ymax, ylim[1])

        return xmin, xmax, ymin, ymax

    # def plot(self, *args, ax=None, **kwargs):
    #     """Plot the map."""
    #     if ax is None:
    #         ax = plt.gca()
    #     ax.plot(self.x_start[0], self.x_start[1], 'ko')
    #     ax.plot(self.x_goal[0], self.x_goal[1], 'r*')
    #     for obs in self.obstacles:
    #         if hasattr(obs, 'plot'):
    #             obs.plot(*args, ax=ax, **kwargs)
    #         else:
    #             print(f"Warning: Obstacle of type {type(obs)} does not have a plot method.")
    #     ax.axis('equal')

    def plot(self, *args, ax=None, **kwargs):
        """Plot the map."""
        if ax is None:
            ax = plt.gca()
        ax.plot(self.x_start[0], self.x_start[1], 'ko')
        ax.plot(self.x_goal[0], self.x_goal[1], 'r*')
        for obs in self.obstacles:
            if hasattr(obs, 'plot'):
                obs.plot(*args, ax=ax, **kwargs)
            else:
                print(f"Warning: Obstacle of type {type(obs)} does not have a plot method.")
        
        ax.set_aspect('equal', 'box')

    def __len__(self):
        return len(self.obstacles)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.collides(np.array(key))
        elif isinstance(key, slice):
            return self.obstacles[key]
        else:
            raise TypeError("Invalid argument type.")
