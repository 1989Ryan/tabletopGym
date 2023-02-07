import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from utils import *
import numpy as np

print(CAMERA_MATRIX)
ray = ray_from_pixel([600, 60])
print(ray)
print(get_focal_lengths(HEIGHT, FOV/180.0*np.pi))
print(camera_coord())
print(CAMERA_QUAT)
print(np.matrix(np.array(CAMERA_ROT).reshape(3,3)))
M = np.matrix(np.array(CAMERA_ROT).reshape(3,3)).T

ray = np.matrix(ray).reshape(-1,1)
ray_star = (np.matrix(ROTATE_90).reshape(3,3) * M * ray)

ray_star[-1] *= -1
ray_star /= abs(ray_star[-1])
print(ray_star)
print(ray_star * DISTANCE * np.sin(-PITCH/180*np.pi) + np.matrix(camera_coord()).reshape(-1,1))
coord = coord_from_pixel([600, 60])
pixel = pixel_from_coord(coord)