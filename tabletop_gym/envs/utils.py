from __future__ import print_function
import os, inspect, sys
import numpy as np
import pybullet as p
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

###################################################
# Constants
###################################################

# config

import random

OBJ_CONFIG_10_B = 'tabletop_gym/envs/config/Objects_conf.json'
OBJ_CONFIG_10_A = 'tabletop_gym/envs/config/Objects_simple_10_ambiguous.json'
OBJ_CONFIG_10 = 'tabletop_gym/envs/config/Objects_simple_10.json'
OBJ_CONFIG_4 = 'tabletop_gym/envs/config/Objects_simple_4.json'
OBJ_CONFIG = 'tabletop_gym/envs/config/Objects_new.json'
ENV_CONFIG = 'tabletop_gym/envs/config/env.json'

class RealSenseD415():
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Set default camera poses.
    front_position = (1.25, -0.1, 2.07)
    front_rotation = (np.pi / 3, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (-0.5, 0.6, 2.07)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (-0.5, -0.6, 2.07)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)

    # Default camera configs.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': front_position,
        'rotation': front_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': left_position,
        'rotation': left_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': right_position,
        'rotation': right_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]

# camera params
LOOK = [0.08, 0.02, 1.07]
DISTANCE = 2.
PITCH = -70
YAW = 90
ROLL = 0
HEIGHT = 640
WIDTH = 640

FOV = 30.0
NEAR = 0.01
FAR = 10
ASPECT = WIDTH/HEIGHT

def get_camera_matrix(width, height, fx, fy=None):
    if fy is None:
        fy = fx
    #cx, cy = width / 2., height / 2.
    cx, cy = (width - 1) / 2., (height - 1) / 2.
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

def get_focal_lengths(dims, fovs):
    return np.divide(dims, np.tan(fovs / 2)) / 2.0

CAMERA_MATRIX = get_camera_matrix(WIDTH, HEIGHT, get_focal_lengths(HEIGHT, FOV/180.0*np.pi))

CAMERA_QUAT = p.getQuaternionFromEuler([0, -(np.pi*(-PITCH))/180.0, 0])
CAMERA_ROT = p.getMatrixFromQuaternion(CAMERA_QUAT)

ROTATE_90 = p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, -np.pi/2, 0]))

def ray_from_pixel(pixel, camera_matrix=CAMERA_MATRIX):
    return np.linalg.inv(camera_matrix).dot(np.append(pixel, 1))

def pixel_from_ray(ray, camera_matrix=CAMERA_MATRIX):
    return camera_matrix.dot(np.array(ray) / ray[2])[:2]

def camera_coord():
    return DISTANCE * np.cos(-PITCH/180*np.pi) + LOOK[0], LOOK[1], DISTANCE * np.sin(-PITCH/180*np.pi) + LOOK[2]

def coord_from_pixel(pixel):
    """transformating pixels to the coordinates in the 3D world"""
    ray = ray_from_pixel(pixel)
    M = np.matrix(np.array(CAMERA_ROT).reshape(3,3)).T
    ray = np.matrix(ray).reshape(-1,1)
    ray_star = (np.matrix(ROTATE_90).reshape(3,3) * M * ray)
    ray_star[-1] *= -1
    ray_star /= abs(ray_star[-1])
    pos = ray_star * DISTANCE * np.sin(-PITCH/180*np.pi) + np.matrix(camera_coord()).reshape(-1,1)
    return pos

def pixel_from_coord(coord):
    """transformating coordinates to pixels in the observation"""
    
    coord = np.matrix(coord).reshape(-1,1)
    ray_star = coord - np.matrix(camera_coord()).reshape(-1,1)
    ray_star = ray_star / (DISTANCE * np.sin(-PITCH/180*np.pi))
    ray_star /= abs(ray_star[-1])
    ray_star[-1] *= -1
    print(ray_star)
    M = np.matrix(np.array(CAMERA_ROT).reshape(3,3)).T
    ray = np.linalg.inv(np.matrix(ROTATE_90).reshape(3,3) * M) * ray_star
    pixel = pixel_from_ray(ray).reshape(1,-1).tolist()[0]
    return int(pixel[0]), int(pixel[1])

# default
DEFAULT_STATE = 14
DEFAULT_TABLE = 407
DEFAULT_STEP = 10

# robot path
FRANKA_URDF = 'tabletop_gym/envs/assets/franka_description/robots/panda_arm_hand.urdf'

# env
SHORT_FLOOR = 'tabletop_gym/envs/assets/objects/short_floor.urdf'
WINE_GLASS = 'tabletop_gym/envs/assets/objects/Wine Glass BB (1).obj'
FLOOR_TEX = 'tabletop_gym/envs/assets/objects/floor.png'
KUKA_URDF = 'tabletop_gym/envs/assets/drake/iiwa_description/urdf/iiwa14_polytope_collision.urdf'
TABLE_OBJ = 'tabletop_gym/envs/assets/objects/table.obj'
TABLE_TEX = 'tabletop_gym/envs/assets/objects/Table_BaseColor.tga'
WATER_GLASS = 'tabletop_gym/envs/assets/objects/Water Glass BB.obj'
CLOTH_TEX = 'tabletop_gym/envs/assets/objects/T_Cotton_Height.tga'
DINNER_FORK = 'tabletop_gym/envs/assets/objects/Dinner Fork BB.obj'
DINNER_KNIFE = 'tabletop_gym/envs/assets/objects/Dinner Knife BB.obj'
SPOON = 'tabletop_gym/envs/assets/objects/Soup Spoon BB.obj'
PLATE = 'tabletop_gym/envs/assets/objects/blue_plate/textured.obj'
PLATE_COL = 'tabletop_gym/envs/assets/objects/blue_plate/collision.obj'
PLATE_TEX = 'tabletop_gym/envs/assets/objects/blue_plate/texture_map.jpg'
NAPKIN = 'tabletop_gym/envs/assets/objects/napkin/Napkin Cloth BB.obj'
COFFIE = 'tabletop_gym/envs/assets/objects/cupa.obj'
UR5_URDF = 'tabletop_gym/envs/assets/ur5/ur5.urdf'
UR5_SUCTION_BASE_URDF = 'tabletop_gym/envs/assets/ur5/suction/suction-base.urdf'
UR5_SUCTION_HEAD_URDF = 'tabletop_gym/envs/assets/ur5/suction/suction-head.urdf'
LIGHT_DIRECTION = [0.3,-0.2, 2.3]
TABLE_HEIGHT = 1.08
TABLE_HEIGHT_PLUS = 1.09

TABLE_CLOTH = 'tabletop_gym/envs/assets/objects/Table cloth.obj'

# render
RENDER_LOOK = [-0.0, 0., 1.2]
RENDER_DISTANCE = 5.5
RENDER_PITCH = -40
RENDER_YAW = 45
RENDER_ROLL = 0
def render_camera_coord():
    return 1.7, 0., 3.5
    # return RENDER_DISTANCE * np.cos(-RENDER_PITCH/180*np.pi) + RENDER_LOOK[0], RENDER_LOOK[1], RENDER_DISTANCE * np.sin(-RENDER_PITCH/180*np.pi) + RENDER_LOOK[2]

# dataset
SCENE_DIR = 'tabletop_gym/envs/scenes/'
OBJECT_OBJ_MAT_CONF = "tabletop_gym/envs/assets/train/config.json"

# color

BRONZE_RGBA = [80./255., 47./255., 32./255., 1]
GOLD_RGBA = [156./255.,124./255.,56./255., 1]
SILVER_RGBA = [-0.08, -0.08, 1.08]
IRON_RGBA = [70./255., 70./255., 73./255., 1]
# BLUE_RGBA = [163, 214, 245, 255] / 255
# YELLOW_RGBA = [244, 214, 188, 255] / 255
# GREEN_RGBA = [50, 205, 50, 255] / 255
# RED_RGBA = [255, 88, 71, 255] / 255
# PURPLE_RGBA = [238, 130, 238, 255] / 255

colors = {
    "bronze" : [80., 47., 32., 255],
    "gold" : [156.,124.,56., 255],
    "grey" : [140., 140., 143, 255],
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
    "lemmon_yellow": [255,250,155, 255],
    "lavender": [255,210,215, 255]
}

colors_name_train = {
    "red" : ["red"],
    "yellow" : ["yellow", "gold"],
    "green" : ["green"],
    "blue" : ["blue", "cyan"],
    "silver": ["grey"],
    "grey": ["grey"],
    "white": ["white"],
    "pure_white": ["pure_white"],
    "sky": ["sky_blue"],
    "lemmon": ["lemmon_yellow"],
    "lavender": ["lavender"],
}

colors_name_test_1 = {
    "red" : ["red"],
    "yellow" : ["yellow", "gold"],
    "green" : ["green"],
    "blue" : ["blue", "cyan"],
    "silver": ["grey"],
    "grey": ["grey"],
    "white": ["white"],
}

colors_name_test_2 = {
    "red" : ["red", "pink"],
    "yellow" : ["orange", "yellow", "gold"],
    "green" : ["green", "dark_green"],
    "blue" : ["blue", "cyan", "sky_blue"],
    "silver": ["grey"],
    "grey": ["grey"],
    "white": ["white"],
}

color_weights = { 
    "mug": {
        "red" : 0.3,
        "yellow" : 0.0,
        "green" : 0.0,
        "blue" : 0.3,
        "grey": 0.3,
        "white": 0.1, 
    },
    "cup": {
        "red" : 0.3,
        "yellow" : 0.0,
        "green" : 0.0,
        "blue" : 0.3,
        "grey": 0.3,
        "white": 0.1, 
    },
    "fork": {
        "red" : 0.0,
        "yellow" : 0.0,
        "green" : 0.0,
        "blue" : 0.0,
        "silver": 1.0,
        "white": 0.0, 
    }, 
    "knife": {
        "red" : 0.0,
        "yellow" : 0.0,
        "green" : 0.0,
        "blue" : 0.0,
        "silver": 1.0,
        "white": 0.0, 
    },  
    "square plate": {
        "red" : 0.0,
        "yellow" : 0.2,
        "green" : 0.0,
        "blue" : 0.4,
        "silver": 0.0,
        "white": 0.4, 
    }, 
    "plate": {
        "red" : 0.0,
        "yellow" : 0.2,
        "green" : 0.0,
        "blue" : 0.4,
        "silver": 0.0,
        "white": 0.4, 
    }, 
    "spoon": {
        "red" : 0.0,
        "yellow" : 0.0,
        "green" : 0.0,
        "blue" : 0.0,
        "silver": 1.0,
        "white": 0.0, 
    }, 
    "table": {
        "red" : 0.0,
        "yellow" : 0.0,
        "green" : 0.0,
        "blue" : 0.0,
        "silver": 0.,
        "white": 0.4, 
        "sky": 0.2,
        "lemmon": 0.2,
        "lavender": 0.2,
    },
    "wine glass": {
        "red" : 0.0,
        "yellow" : 0.0,
        "green" : 0.0,
        "blue" : 0.0,
        "silver": 0.0,
        "white": 1.0, 
    },
}


material_weights = {  
    "mug": {
        # "glass" : 0.3,
        "plastic" : 0.5,
        "metallic" : 0.5,
    }, 

    "cup": {
        # "glass" : 0.3,
        "plastic" : 0.5,
        "metallic" : 0.5,
    }, 
    "fork": {
        "plastic" : 0.0,
        "metallic" : 1.0,
    }, 
    "knife": {
        "plastic" : 0.0,
        "metallic" : 1.0,
    },     
    "square plate": {
        "plastic" : 1.0,
        "metallic" : 0.0,
    }, 
    "plate": {
        "plastic" : 1.0,
        "metallic" : 0.0,
    }, 
    "spoon": {
        "plastic" : 0.0,
        "metallic" : 1.0,
    }, 
    "wine glass": {
        "plastic" : 0.0,
        "metallic" : 1.0,
    }, 
    "table": {
        "plastic": 1.0,
        "metallic": 0.0
    }
}
###################################################
# tools in simulation
###################################################

# string to pybullet.shapeType


SHAPETYPE = {
    'mesh': p.GEOM_MESH,
    'box': p.GEOM_BOX,
}

def str2shapetype(str):
    assert str in SHAPETYPE, ('invalid shapetype \'{}\''.format(str))
    return SHAPETYPE[str]

# read csv

import csv

def read_action_csv(csvpath):
    state = {}
    try:
        with open(csvpath, mode='r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                obj = row[0]
                '''
                referred to the script MakePictures.cs
                line 188-195
                '''
                x = float(row[1])
                y = - float(row[3])
                theta = float(row[5])
                state[obj] = np.array([x, y, theta])
    except IOError as exc:
        raise IOError("%s: %s" % (csvpath, exc.strerror))
    return state

# read and save json

import json

def read_json(filepath):
    '''
    from filepath to instruction list
    :return:instruction list
    '''
    try:
        with open(filepath) as f:
            data = json.load(f)
    except IOError as exc:
        raise IOError("%s: %s" % (filepath, exc.strerror))
    return data

ROTATION_DICT = read_json(CURRENT_DIR + '/config/Rotation.json')
TRANS_DICT = read_json(CURRENT_DIR + '/config/Trans.json')
DEFAULT_CONF_PATH = CURRENT_DIR + '/config/default.csv'

def dump_json(dict_data, filepath):
    with open(filepath, 'w') as fp:
        json.dump(dict_data, fp, indent=2)
        # fp.write('\n')

# check whether the object orientation is enabled in evaluation
    
CHECKLIST = ['Fork', 'Knife', 'Spoon']
def check_orn_enable(obj_name):
    # TODO: make it more efficient
    return any([ele in obj_name for ele in CHECKLIST])


# transformation of two coord.

from scipy.spatial.transform import Rotation

def getMatrixFromEuler(euler):
    r = Rotation.from_euler(euler)
    return np.matrix(r.as_matrix())

def getQuaternionFromMatrix(matrix):
    r = Rotation.from_matrix(matrix)
    return r.as_quat()

def getMatrixFromQuaternion(q):
    r = Rotation.from_quat(q)
    return np.matrix(r.as_matrix())

def getEulerFromMatrix(matrix):
    r = Rotation.from_matrix(matrix)
    return r.as_euler('xyz')

MESH_OBJ_LIST = [
    "small_plate",
    "fork",
    "knife",
    "coffee_cup",
    "table_cloth",
    "napkin_cloth",
    "spoon",
    "table",
    "wine_glass",
    "water_glass"
]

def transform(name, conf, mesh_name):
    position = np.matrix([conf[0], conf[1]]).reshape(-1, 1)
    orientation = conf[2]
    quat = p.getQuaternionFromEuler([0., 0., orientation * np.pi/180.])
    matrix = np.matrix(p.getMatrixFromQuaternion(quat)).reshape(3,3)
    if mesh_name in MESH_OBJ_LIST:
        matrix = np.matrix(ROTATION_DICT[name]) * matrix
    orn = getQuaternionFromMatrix(matrix)
    pos = np.matrix(TRANS_DICT['A']) * position + np.matrix(TRANS_DICT['b'])
    pos = pos.reshape(1,-1).tolist()[0] + [TABLE_HEIGHT]
    return pos, orn


# def transform(name, conf):
#     position = np.matrix([conf[0], conf[1]]).reshape(-1, 1)
#     orientation = conf[2]
#     quat = p.getQuaternionFromEuler([0., 0., orientation * np.pi/180.])
#     matrix = np.matrix(p.getMatrixFromQuaternion(quat)).reshape(3,3)
#     matrix = np.matrix(ROTATION_DICT[name]) * matrix
#     orn = getQuaternionFromMatrix(matrix)
#     pos = np.matrix(TRANS_DICT['A']) * position + np.matrix(TRANS_DICT['b'])
#     pos = pos.reshape(1,-1).tolist()[0] + [TABLE_HEIGHT]
#     return pos, orn

# def transform(name, conf):
#     position = np.matrix([conf[0], conf[1]]).reshape(-1, 1)
#     orientation = conf[2]
#     quat = p.getQuaternionFromEuler([0., 0., orientation * np.pi/180.])
#     matrix = np.matrix(p.getMatrixFromQuaternion(quat)).reshape(3,3)
#     matrix = np.matrix(ROTATION_DICT[name]) * matrix
#     random_quat = p.getQuaternionFromEuler([0., 0., orientation * np.pi/random.uniform(0, 180)])
#     random_matrix = np.matrix(p.getMatrixFromQuaternion(random_quat)).reshape(3,3)
#     matrix = matrix * random_matrix
#     orn = getQuaternionFromMatrix(matrix)
#     pos = np.matrix(TRANS_DICT['A']) * position + np.matrix(TRANS_DICT['b'])
#     pos = pos.reshape(1,-1).tolist()[0] + [TABLE_HEIGHT]
#     return pos, orn

def inv_transform(name, pos, orn, mesh_name):
    if not isinstance(pos, list):
        pos = list(pos)
    pos = pos[:-1]
    pos = np.matrix(pos).reshape(-1, 1)
    position = np.linalg.inv(np.matrix(TRANS_DICT['A']))*(pos - np.matrix(TRANS_DICT['b']))
    matrix = getMatrixFromQuaternion(orn)
    if mesh_name in MESH_OBJ_LIST:
        matrix = np.linalg.inv(np.matrix(ROTATION_DICT[name])) * matrix
    euler = getEulerFromMatrix(matrix)
    euler = euler[-1] / np.pi * 180
    return position, euler



# get csv filename

CSV_DIR = 'tabletop_gym/envs/data/'

def get_csv_path(mode, state_id, step_id, table_id):
    return CSV_DIR + mode + '/HIT_' + str(state_id) + '_configs_' +\
         str(table_id) + '_' + str(step_id) + '.csv'

def parse_csv_filename(filename):
    try:
        '''
        optimize the code
        '''
        # print(filename)
        name = filename[:-4]
        name_list = name.split('_')
        name_list.remove('HIT')
        name_list.remove('configs')
        state_id = int(name_list[-3])
        table_id = int(name_list[-2])
        step_id = int(name_list[-1])
    except ValueError:
        print("value error in transfering string %s" % filename)
    # print(table_idx, idx)
    return state_id, table_id, step_id

# get json filename

JSON_DIR = 'tabletop_gym/envs/data/instruction/'

SCENE_JSON_DIR = 'tabletop_gym/envs/data/instruction/'

def get_json_path(mode, state_id, step_id, table_id):
    return JSON_DIR + mode + '/' + str(state_id) + '_configs_' +\
         str(table_id) +'_ins.json'

def get_json_path_from_scene(mode, state_id, step_id, table_id):
    return SCENE_DIR + mode + '/' + str(state_id) + '_' +\
         str(table_id) +'_' + str(step_id) +'/info.json'

from PIL import Image

def save_images(img, filepath):
    img = Image.fromarray(img)
    img.save(filepath)


########################################################
# tools in ompl
########################################################

'''
Adopted from
https://github.com/StanfordVL/iGibson/blob/master/igibson/external/pybullet_tools/utils.py
'''


from collections import namedtuple
from itertools import product, combinations


INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 5.0

BASE_LINK = -1
MAX_DISTANCE = 0.

def pairwise_link_collision(client, body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE):  # 10000
    return len(client.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2)) != 0  # getContactPoints

def pairwise_collision(client, body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(client, body1)
        body2, links2 = expand_links(client, body2)
        return any_link_pair_collision(client, body1, links1, body2, links2, **kwargs)
    return body_collision(client, body1, body2, **kwargs)

def expand_links(client, body):
    body, links = body if isinstance(body, tuple) else (body, None)
    if links is None:
        links = get_all_links(client, body)
    return body, links

def any_link_pair_collision(client, body1, links1, body2, links2=None, **kwargs):
    # TODO: this likely isn't needed anymore
    if links1 is None:
        links1 = get_all_links(client, body1)
    if links2 is None:
        links2 = get_all_links(client, body2)
    for link1, link2 in product(links1, links2):
        if (client, body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(client, body1, link1, body2, link2, **kwargs):
            # print('body {} link {} body {} link {}'.format(client, body1, link1, body2, link2))
            return True
    return False

def body_collision(client, body1, body2, max_distance=MAX_DISTANCE):  # 10000
    return len(client.getClosestPoints(client, bodyA=body1, bodyB=body2, distance=max_distance)) != 0  # getContactPoints`

def get_self_link_pairs(client, body, joints, disabled_collisions=set(), only_moving=True):
    moving_links = get_moving_links(client, body, joints)
    fixed_links = list(set(get_joints(client, body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(client, body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))
    check_link_pairs = list(
        filter(lambda pair: not are_links_adjacent(client, body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def get_moving_links(client, body, joints):
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(client, body, link))
    return list(moving_links)

def get_moving_pairs(client, body, moving_joints):
    """
    Check all fixed and moving pairs
    Do not check all fixed and fixed pairs
    Check all moving pairs with a common
    """
    moving_links = get_moving_links(client, body, moving_joints)
    for link1, link2 in combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(client, body, link1)) & set(moving_joints)
        ancestors2 = set(get_joint_ancestors(client, body, link2)) & set(moving_joints)
        if ancestors1 != ancestors2:
            yield link1, link2

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

def get_joint_info(client, body, joint):
    return JointInfo(*client.getJointInfo(body, joint))

def child_link_from_joint(joint):
    return joint  # link

def get_num_joints(client, body):
    return client.getNumJoints(body)

def get_joints(client, body):
    return list(range(get_num_joints(client, body)))

get_links = get_joints

def get_all_links(client, body):
    return [BASE_LINK] + list(get_links(client, body))

def get_link_parent(client, body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(client, body, link).parentIndex

def get_all_link_parents(client, body):
    return {link: get_link_parent(client, body, link) for link in get_links(client, body)}

def get_all_link_children(client, body):
    children = {}
    for child, parent in get_all_link_parents(client, body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children

def get_link_children(client, body, link):
    children = get_all_link_children(client, body)
    return children.get(link, [])


def get_link_ancestors(client, body, link):
    # Returns in order of depth
    # Does not include link
    parent = get_link_parent(client, body, link)
    if parent is None:
        return []
    return get_link_ancestors(client, body, parent) + [parent]


def get_joint_ancestors(client, body, joint):
    link = child_link_from_joint(joint)
    return get_link_ancestors(client, body, link) + [link]

def get_link_descendants(client, body, link, test=lambda l: True):
    descendants = []
    for child in get_link_children(client, body, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(client, body, child, test=test))
    return descendants


def get_link_subtree(client, body, link, **kwargs):
    return [link] + get_link_descendants(client, body, link, **kwargs)

def are_links_adjacent(client, body, link1, link2):
    return (get_link_parent(client, body, link1) == link2) or \
           (get_link_parent(client, body, link2) == link1)


##########################################################
# tools in TAMP solver
##########################################################

# tamp problem definition model

DiscreteTAMPState = namedtuple('DiscreteTAMPState', ['conf', 'holding', 'object_poses'])
DiscreteTAMPProblem = namedtuple('DiscreteTAMPProblem', ['initial', 'poses', 'goal_poses'])

INITIAL_CONF = ((0.487-1, 0.109+0.6, 0.347+1.07236), (0.0, 0.0, 0.0, 1.0)) 
# TODO: change the initial configuration

def distance_fn(q1, q2):
    return np.linalg.norm(np.array(q1[0])[:2] - np.array(q2[0])[:2], ord=1) + \
        np.linalg.norm(np.array(q1[1]) - np.array(q2[1]), ord=1)

def get_shift_all_problem(initial_object_poses, goal_object_poses, temp_poses=[]):
    print("enter my shift all problem")
    initial = DiscreteTAMPState(INITIAL_CONF, None, initial_object_poses)
    # print(goal_object_poses.values())
    available_poses = list(goal_object_poses.values()) + temp_poses

    return DiscreteTAMPProblem(initial, available_poses, goal_object_poses)

# motion.py

def get_diff_quat(q1, q2):
    return p.getDifferenceQuaternion(q1, q2)



##########################################################
# UR5 utils
##########################################################

LEAVE_HEIGHT = 0.4

from transforms3d import euler

def multiply(pose0, pose1):
    return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])

def eulerXYZ_to_quatXYZW(rotation):  # pylint: disable=invalid-name
    """Abstraction for converting from a 3-parameter rotation to quaterion.
    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.
    Args:
        rotation: a 3-parameter rotation, in xyz order tuple of 3 floats
    Returns:
        quaternion, in xyzw order, tuple of 4 floats
    """
    euler_zxy = (rotation[2], rotation[0], rotation[1])
    quaternion_wxyz = euler.euler2quat(*euler_zxy, axes='szxy')
    q = quaternion_wxyz
    quaternion_xyzw = (q[1], q[2], q[3], q[0])
    return quaternion_xyzw

def get_euler_from_quat(q):
    return p.getEulerFromQuaternion(q)

############### get bounding boxes from masks #################

from PIL import ImageDraw

def get_segmentation_mask_object_and_link_index(seg_image):
    """
    Following example from
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py
    :param seg_image: [H, W] segmentation bitmap
    :return: object id map
    :return: link id map
    """
    assert(seg_image.ndim == 2)
    bmask = seg_image >= 0
    obj_idmap = bmask.copy().astype(np.int64) - 1
    link_idmap = obj_idmap.copy() - 1
    obj_idmap[bmask] = seg_image[bmask] & ((1 << 24) - 1)
    link_idmap[bmask] = (seg_image[bmask] >> 24) - 1
    return obj_idmap, link_idmap

def get_bbox2d_from_mask(bmask):
    """
    Get 2d bbox from a binary segmentation mask
    :param bmask: binary segmentation mask
    :return:
    """
    box = np.zeros(4, dtype=np.int64)
    coords_r, coords_c = np.where(bmask > 0)
    if len(coords_r) == 0:
        print('WARNING: empty bbox')
        return box
    box[0] = coords_r.min()
    box[1] = coords_c.min()
    box[2] = coords_r.max()
    box[3] = coords_c.max()
    return box

def union_boxes(boxes):
    """
    Union a list of boxes
    :param boxes: [N, 4] boxes
    :return:
    """
    assert(isinstance(boxes, (tuple, list)))
    boxes = np.vstack(boxes)
    new_box = boxes[0].copy()
    new_box[0] = boxes[:, 0].min()
    new_box[1] = boxes[:, 1].min()
    new_box[2] = boxes[:, 2].max()
    new_box[3] = boxes[:, 3].max()
    return new_box


def get_bbox2d_from_segmentation(seg_map, object_ids):
    """
    Get 2D bbox from a semantic segmentation map
    :param seg_map:
    :return:
    """
    all_bboxes = np.zeros([len(object_ids), 5], dtype=np.int64)
    for i in range(len(object_ids)):
        all_bboxes[i, 0] = object_ids[i]
        all_bboxes[i, 1:] = get_bbox2d_from_mask(seg_map == object_ids[i])
    return all_bboxes


def box_rc_to_xy(box):
    """
    box coordinate from (r1, c1, r2, c2) to (x1, y1, x2, y2)
    :param box: a
    :return: box
    """
    return np.array([box[1], box[0], box[3], box[2]], dtype=box.dtype)

def draw_boxes(image, boxes, labels=None):
    if labels is not None:
        assert(len(labels) == len(boxes))
    image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(image)
    for b in boxes:
        draw.rectangle(b.tolist(), outline='green')
    return np.array(image)