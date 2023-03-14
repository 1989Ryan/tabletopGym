import tabletop_gym.envs.object as obj_dict
import pdb
import numpy as np
import random
import math
import itertools


object_types = [
    'cup', 'knife', 'spoon', 'fork', 'plate', 'napkin'
]

object_dict = {'cup': ['cup1', 'cup2', 'cup3', 'cup4', 'cup5'], 
    'knife': ["knife1"],
    'spoon': ["spoon1"],
    'fork': ["fork1"],
    'plate': ["plate1", "plate2", "plate3"],
    'napkin': ["napkin"]}

colors = {
    "bronze" : [80., 47., 32., 255],
    "gold" : [156.,124.,56., 255],
    "silver" : [140., 140., 143, 255],
    "blue" : [53, 104, 195, 255],
    "yellow" : [244, 214, 60, 255] ,
    "green" : [50, 205, 50, 255] ,
    "red" : [255, 88, 71, 255] ,
    "purple" : [100, 104, 200, 255],
    "cyan": [94, 185, 255, 255],
    "orange": [255, 165, 0, 255] ,
    "brown": [165, 42, 42, 255] ,
    "pink": [240, 128, 128, 255] ,
    "dark_green": [0, 128, 0, 255] ,
    "sky_blue": [135, 206, 250, 255] ,
    "white": [200, 200, 180, 255] ,
    "pure_white": [240, 240, 240, 255] ,
    "lemon_yellow": [255,250,155, 255],
    "lavender": [255,210,215, 255],
}

material = ['plastic', 'metallic', 'cloth']

def visualize_all():
    object_list = []

    pos_count = 0
    obj_idx = 0
    for obj_type_ in object_dict.keys():
        for obj_type in object_dict[obj_type_]:
            if obj_type == "cup1":
                for c in colors.keys():
                    color = colors[c]
                    x, y = pos_count%8*4, pos_count//8*4
                    angle = pos_count%4

                    o = {"type": obj_type, "name": len(object_list), "color": color, "pos": (x, y), "angle": angle}
                    object_list.append(o)
                    pos_count += 1
            else:
                default_color = obj_dict.texture_color[obj_type]
                if default_color is None:
                    color = colors["white"]
                else:
                    color = colors[default_color]
                x, y = pos_count%8*4, pos_count//8*4
                angle = pos_count%4
                o = {"type": obj_type, "name": len(object_list), "color": color, "pos": (x, y), "angle": angle}
                object_list.append(o)
                pos_count += 1
    pdb.set_trace()
    return object_list    

def get_tidy_config(num_cup1=25, num_cup2=10, num_uten=12, horizontal=True, stick_to_left=True):

    grid = np.zeros((32, 32), dtype=int)
    # num_napkin_list = [0, 1, 2, 3]

    type_list = ["cup1", "cup2", "fork1"]
    cluster_size = {}

    cluster_size["cup1"] = (int(np.sqrt(num_cup1))*3, int(np.sqrt(num_cup1))*3)
    cluster_size["fork1"] = (num_uten, 8)
    if horizontal:
        cluster_size["cup2"] = (min(num_cup2, 5)*3, math.ceil(num_cup2/5)*3)        
    else: 
        cluster_size["cup2"] = (math.ceil(num_cup2/5)*3, min(num_cup2, 5)*3)
        # cluster_size["fork1"] = (10, num_uten*4)

    left_start_idx = (0, 0)
    right_start_idx = (32, 0)
    random.shuffle(type_list)

    obj_cluster_pos = {}
    for idx, obj in enumerate(type_list):
        if idx == 0:
            x0, y0 = 0, 0
            x1, y1 = cluster_size[obj]
            if horizontal:
                left_start_idx = (0, y1)
            else:
                left_start_idx = (x1, 0)
        elif idx == 1:
            _x, _y = cluster_size[obj]
            x0, y0 = (32 - _x, 0)
            x1, y1 = (32, _y)
            if horizontal:
                right_start_idx = (32, y1)
            else:
                right_start_idx = (x0, 0)
        else:
            if stick_to_left:
                x0, y0 = left_start_idx
                x1, y1 = x0 + cluster_size[obj][0], y0 + cluster_size[obj][1]
            else:
                _x, _y = cluster_size[obj]
                x0, y0 = right_start_idx[0] - _x, right_start_idx[1]
                x1, y1 = right_start_idx[0], _y + right_start_idx[1]
        
        obj_cluster_pos[obj] = (x0, y0, x1, y1) 

    type_list = ["cup1", "cup2", "fork1"]

    obj_list = []
    for _type in type_list:
        if _type == "cup1":
            color_list = np.random.choice(list(colors.keys()), num_cup1)
            x_0, y_0, x_1, y_1 = obj_cluster_pos[_type]
            k = int(np.sqrt(num_cup1))
            for i in range(num_cup1):
                color = colors[color_list[i]]
                x, y = x_0 + i%k*3, y_0 + i//k*3
                angle = 0
                o = {"type": _type, "name": len(obj_list), "color": color, "pos": (x, y), "shape": (3, 3), "angle": angle}
                obj_list.append(o)
                grid[x:x+3, y:y+3] = len(obj_list)
        elif _type == "cup2": 
            color = colors["white"]
            x_0, y_0, x_1, y_1 = obj_cluster_pos[_type]
            for i in range(num_cup2):
                if horizontal:
                    x, y = x_0 + i%5*3, y_0 + i//5*3
                else:
                    x, y = x_0 + i//5*3, y_0 + i%5*3
                angle = 0
                o = {"type": _type, "name": len(obj_list), "color": color, "pos": (x, y), "shape": (3, 3), "angle": angle}
                obj_list.append(o)
                grid[x:x+3, y:y+3] = len(obj_list)
        elif _type == "fork1":
            color = colors["silver"]
            x_0, y_0, x_1, y_1 = obj_cluster_pos[_type]
            for i in range(num_uten):
                x, y = x_0 + i, y_0
                angle = 0
                grid[x:x+1, y:y+8] = len(obj_list)
                # if horizontal:
                #     x, y = x_0 + i, y_0
                #     angle = 0
                #     grid[x:x+1, y:y+10] = len(obj_list)
                # else:
                #     x, y = x_0, y_0 + i*4
                #     angle = 90
                #     grid[x:x+10, y:y+4] = len(obj_list)
                o = {"type": _type, "name": len(obj_list), "color": color, "pos": (x, y), "shape": (1, 8), "angle": angle}
                obj_list.append(o)
        else:
            print("No such object!")

    return obj_list, grid

def batch_initialize_tidy_config():

    num_cup1_list = [4, 9, 16]
    num_cup2_list = np.arange(4, 15).tolist()
    num_utensil_list = np.arange(4, 10).tolist()
    horizontal_list = [True, False]
    stick_to_left_list = [True, False]

    batch_tidy_config = []

    for element in itertools.product(num_cup1_list, num_cup2_list, num_utensil_list, horizontal_list, stick_to_left_list):
        num_cup1, num_cup2, num_uten, horizontal, stick_to_left = element
        obj_list, grid = get_tidy_config(num_cup1, num_cup2, num_uten, horizontal, stick_to_left)
        batch_tidy_config.append((obj_list, grid.tolist(), element))

    return batch_tidy_config


def random_walk_one_step(obj, grid):
    w, h = obj["shape"]
    x0, y0 = obj["pos"]
    while True:
        x, y = random.sample(range(32), 2)
        if np.all(grid[x:x+w, y:y+h]) == 0:
            grid[x:x+w, y:y+h] = obj["name"]
            grid[x0:x0+w, y0:y0+h] = 0
            obj["pos"] = (x, y)
            break 

    return obj, grid

    


    

