import math
import numpy as np

def cm_to_m(x, dec = 4):
    return round(x/100,dec)

def array_cm_to_m(array):
    n = np.ndim(array)
    if n == 1:
        res = [cm_to_m(x) for x in array]
    elif n == 2:
        res = [[cm_to_m(y) for y in x] for x in array]
    else:
        print("Conversion not implemented.")
    return res

def array_deg_to_rad(array):
    return [math.radians(x) for x in array]

def array_rad_to_deg(array):
    return [math.degrees(x) for x in array]

# object poses in cm
object_positions = [

    [-57.5, 6.3, -2.78],            # 1
    [-40, 6.3, -2.78],              # 2
    [-22.8, 6.3, -2.78],            # 3
    [-14.15, -19.3, -1.67],         # 4
    [13.1, -40.5, 0.2],             # 5
    [1.6, -35, -0.68],              # 6
    [3.85, -54.7, 0.14],            # 7
    [-8.7, -44.7, 2.92],            # 8
    [-42, -39.5, 0.13]              # 9

] # in cm

delivery_pose = [-42, -39.5, 0.13] # in cm

tool_orientation = [0, 180, 0] # in deg

home_joints = [20.3, -117.4, 113.2, -84.5, -90.6, -20.6] # in deg

x_axis_offset = [0,0,0,10,0,0,0,0,20] # in cm

PICK_HEIGHT = 8 # in cm