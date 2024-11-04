from map import Map
from quadratic_obstacle import QuadraticObstacle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import utils

class RegionMap(Map):
    def __init__(self):
        super().__init__()
        self.regions: Dict[str, Dict[str, Any]] = {}
        self.map_version = 'v1'

    def add_obstacle(self, obstacle: QuadraticObstacle):
        """Add one obstacle (infeasible region) to the map."""
        self.add(obstacle)

    def add_obstacles(self, *obstacles):
        """Add multiple obstacles."""
        self.add(*obstacles)

    def new_region(self, name: str, color):
        """
        Create a new region.

        Args:
            name: Name of the region (must be alphanumeric and not start with a number).
            color: Color for plotting the region.
        """
        if self.region_exists(name):
            raise ValueError(f"Name '{name}' already in use for areas")
        self.regions[name] = {'shapes': [], 'color': utils.color2RGB(color)}

    def add_shape_to_region(self, region: str, obstacle: QuadraticObstacle):
        """
        Add one shape (obstacle) to a region.

        Args:
            region: Name of the region.
            obstacle: QuadraticObstacle object to add to the region.
        """
        if not self.region_exists(region):
            raise ValueError(f"Unknown type '{region}' of penalty obstacles. Use new_region method to define it")
        assert isinstance(obstacle, QuadraticObstacle)
        self.regions[region]['shapes'].append(obstacle)

    def add_shapes_to_region(self, region: str, *obstacles):
        """
        Add multiple shapes (obstacles) to a region.

        Args:
            region: Name of the region.
            *obstacles: QuadraticObstacle objects to add to the region.
        """
        for obstacle in obstacles:
            self.add_shape_to_region(region, obstacle)

    # return ex: ['HistCenter', 'Population', 'Land']
    def region_names(self) -> List[str]:
        """Get the names of all regions."""
        return list(self.regions.keys())

    def region_exists(self, region: str) -> bool:
        """Check if a region exists."""
        return region in self.regions

    def get_axislim(self) -> Tuple[float, float, float, float]:
        """Get the axis limits for plotting."""
        xmin, xmax, ymin, ymax = super().get_axislim()
        
        # names = self.region_names() # ['HistCenter', 'Population', 'Land']
        for region in self.regions.values():
            # print(region['shapes'])
            for obs in region['shapes']:
                xlim = obs.xlim
                ylim = obs.ylim
                xmin = min(xmin, xlim[0])
                xmax = max(xmax, xlim[1])
                ymin = min(ymin, ylim[0])
                ymax = max(ymax, ylim[1])

        return xmin, xmax, ymin, ymax

    def plot(self, *args, ax=None, **kwargs):
        """Plot the map with regions."""
        if ax is None:
            ax = plt.gca()
        for name, region in self.regions.items():
            # print(region['shapes'])
            color = region['color']
            for shape in region['shapes']:
                shape.plot(color=color)
        legend_elements = []
        for name, region in self.regions.items():
            color = region['color']
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=name,
                                              markerfacecolor=color, markersize=10))
        super().plot(*args, ax=ax, **kwargs)
        ax.legend(handles=legend_elements)
        ax.autoscale_view()

if __name__ == '__main__':
    from polygon import polygon
    map = RegionMap()
    map.x_start = np.array([15, -30])
    map.x_goal = np.array([40, -5])
    points = [
        [16.088709677419356, 11.006493506493506],
        [12.21774193548387, -7.8246753246753284],
        [28.245967741935484, -27.629870129870138],
        [33.20564516129032, -16.83441558441559],
        [28.48790322580645, 1.9967532467532438]
    ]

    points2 = [
        [25.04032258064516, -24.464285714285722],
        [33.931451612903224, -38.26298701298702],
        [48.14516129032258, -22.43506493506494],
        [34.596774193548384, -12.207792207792211]
    ]
    obs1 = polygon(*points)
    obs2 = polygon(*points2)
    obstacles = [obs1, obs2]
    map.add_obstacles(*obstacles)
    map.new_region('Region1', 'r')
    map.add_shape_to_region('Region1', obs1)
    map.new_region('Region2', 'b')
    map.add_shape_to_region('Region2', obs2)
    map.plot()
    plt.show()
