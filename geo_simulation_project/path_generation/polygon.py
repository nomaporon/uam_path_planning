import numpy as np
from function import Function
from typing import List, Tuple, Optional
from quadratic_obstacle import QuadraticObstacle
import casadi.casadi as cs

def polygon(*points) -> QuadraticObstacle:
    """
    Create a convex polygon delimited by points P1, P2, P3, ...
    
    Args:
        *points: List of 2D points defining the polygon vertices
        
    Returns:
        QuadraticObstacle: A quadratic obstacle representing the polygon
        
    Raises:
        ValueError: If less than 3 points are given or if points are not properly formatted
    """
    if len(points) < 3:
        raise ValueError(f'Only {len(points)} vertices given. At least 3 required')

    # Convert all points to numpy arrays and ensure they are 2D column vectors
    points = [np.array(p).reshape(2, 1) for p in points]
    N = len(points)
    
    P1 = points[0]
    Pa = P1
    
    # Calculate initial area and center
    area = 0
    center = Pa.copy()
    for b in range(1, N):
        Pb = points[b]
        if Pb.shape != (2, 1):
            raise ValueError(f'P_{b+1} size {Pb.shape} different from P_1 size (2,1)')
        center += Pb

    # Create obstacle
    obs = QuadraticObstacle()
    
    # Initialize tracking variables
    indices = np.zeros(N)
    indices[0] = 1
    remaining = list(range(1, N))
    a = 0  # Using 0-based indexing
    n = 1
    
    # Track bounds
    xmin = Pa[0, 0]
    xmax = Pa[0, 0]
    ymin = Pa[1, 0]
    ymax = Pa[1, 0]

    def are_consecutive(a_F: int, b_F: int) -> Tuple[bool, Optional[Function]]:
        """
        Check if two points can be consecutive vertices in the convex polygon.
        
        Args:
            a_F: Index of first point
            b_F: Index of second point
            
        Returns:
            Tuple of (is_consecutive, function) where function defines the edge if consecutive
        """
        Pa_F = points[a_F]
        Pb_F = points[b_F]
        
        def line_F(x):
            return ((Pb_F[1, 0] - Pa_F[1, 0]) * (x[0] - Pa_F[0, 0]) - 
                   (Pb_F[0, 0] - Pa_F[0, 0]) * (x[1] - Pa_F[1, 0]))
        
        sgn_F = 0
        
        # Check all other points
        for j_F in range(N):
            if j_F == a_F or j_F == b_F:
                continue
                
            sgn1_F = np.sign(line_F(points[j_F]))
            if sgn1_F == 0:
                raise ValueError('Input contains three aligned points')
                
            if sgn_F == 0:
                sgn_F = sgn1_F
                continue
                
            if sgn1_F != sgn_F:
                return False, None
        
        if sgn_F == 0:
            raise ValueError('The polygon is nonconvex')
            
        # Create function for the edge
        grad_F = np.array([[Pb_F[1, 0] - Pa_F[1, 0]], 
                          [Pb_F[0, 0] - Pa_F[0, 0]]])
        
        f_F = Function(lambda x: -sgn_F * line_F(x),
                      lambda x: -sgn_F * grad_F,
                      np.zeros((2, 2)))
        
        return True, f_F

    # Main loop to construct the polygon
    while remaining:
        n += 1
        # Find n-th vertex
        found_next = False
        for i, b in enumerate(remaining):
            test, f = are_consecutive(a, b)
            if test:
                Pb = points[b]
                xmin = min(xmin, Pb[0, 0])
                xmax = max(xmax, Pb[0, 0])
                ymin = min(ymin, Pb[1, 0])
                ymax = max(ymax, Pb[1, 0])
                
                indices[n-1] = b + 1  # +1 to match MATLAB 1-based indexing
                remaining.pop(i)
                area += Pa[0, 0] * Pb[1, 0] - Pa[1, 0] * Pb[0, 0]
                a = b
                Pa = Pb
                obs.add(f)
                found_next = True
                break
                
        if not found_next:
            raise ValueError('The polygon is nonconvex')

    # Close the polygon
    test, f = are_consecutive(a, 0)  # Connect to first point
    if not test:
        raise ValueError("Couldn't close polygon")
        
    area += Pa[0, 0] * P1[1, 0] - Pa[1, 0] * P1[0, 0]
    obs.add(f)
    
    # Set obstacle properties
    obs.set_plot_data([xmin, xmax], [ymin, ymax])
    obs.area = abs(area) / 2
    obs.center = center / N
    
    return obs

# Example usage:
if __name__ == "__main__":
    # Create a square
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

    try:
        obstacle = polygon(*points)
        obstacle2 = polygon(*points2)

        import matplotlib.pyplot as plt
        obstacle.plot()
        obstacle2.plot()
        plt.axis('equal')
        plt.xlim(10, 50)
        plt.ylim(-40, 15)
        plt.grid(True)
        plt.show()
        
    except ValueError as e:
        print(f"Error creating polygon: {e}")