from numpy import testing
from src.helper import generate_scan_poses
import math
import numpy as np

# testing data
r = 0.06
z = 0.5
theta_start = math.radians(120)
theta_end = math.radians(60)
cyl_points = np.array([[r,theta_start, z], [r,theta_end, z], [r,theta_end, z+1]]) 
triggers = np.array([1,0,0])

# testing function 
traces  = generate_scan_poses(cyl_points, triggers)

print(traces)