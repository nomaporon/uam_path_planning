import numpy as np
from typing import Union, List, Tuple, Optional
from function import Function
from quadratic_obstacle import QuadraticObstacle

def square(center: Union[List, np.ndarray], r1: float, r2: Optional[float] = None) -> QuadraticObstacle:
    """
    Create a square (or rectangular) obstacle centered at <center> with sides 2*r1 and 2*r2.
    
    Args:
        center: Center point of the square [x, y]
        r1: Half-width in x-direction
        r2: Half-height in y-direction (optional, defaults to r1)
        
    Returns:
        QuadraticObstacle: A square/rectangular obstacle
        
    Raises:
        ValueError: If center is not a 2D point
    """
    # Convert center to numpy array if it isn't already
    center = np.array(center).reshape(2)
    
    # If r2 is not provided, make it equal to r1 (square)
    if r2 is None:
        r2 = r1
    
    # Define the four sides of the square/rectangle
    right = Function(
        lambda x: x[0] - center[0] - r1,
        lambda x: np.array([1.0, 0.0]),
        np.zeros((2, 2))
    )
    
    left = Function(
        lambda x: -x[0] + center[0] - r1,
        lambda x: np.array([-1.0, 0.0]),
        np.zeros((2, 2))
    )
    
    top = Function(
        lambda x: x[1] - center[1] - r2,
        lambda x: np.array([0.0, 1.0]),
        np.zeros((2, 2))
    )
    
    bottom = Function(
        lambda x: -x[1] + center[1] - r2,
        lambda x: np.array([0.0, -1.0]),
        np.zeros((2, 2))
    )
    
    # Create the obstacle from the four sides
    obs = QuadraticObstacle(right, left, top, bottom)
    
    # Set plot data and other properties
    obs.set_plot_data(
        [center[0] - r1, center[0] + r1],
        [center[1] - r2, center[1] + r2]
    )
    
    obs.center = center
    obs.area = 4 * r1 * r2
    
    return obs

# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    square_obs = square([1, 1], 0.5)
    rect_obs = square([0, 0], 1.0, 0.5)
    
    plt.figure(figsize=(8, 6))
    
    square_obs.plot(color='blue', label='Square')
    rect_obs.plot(color='red', label='Rectangle')
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    plt.show()