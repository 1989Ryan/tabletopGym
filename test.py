from tabletop_gym.envs.pb_env_nvisii import Tabletop_Sim
import tabletop_gym.envs.object as obj_dict

object_types = [
    'cup', 'kinfe', 'spoon', 'fork', 'plate', 'napkin'
]

object_lists = [
    'cup1', 
    'cup2', 
    'cup3', 
    'cup4', 
    'cup5', 
    'knife1', 
    'spoon1', 
    'fork1' ,
    'plate1', 
    'plate2', 
    'plate3', 
    'plate4', 
    'napkin', 
]

colors = {
    "bronze" : [80., 47., 32., 255],
    "gold" : [156.,124.,56., 255],
    "sliver" : [140., 140., 143, 255],
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

material = ['plastic', 'metallic', 'cloth']

sim = Tabletop_Sim(
    width=640,
    height=640,
    indivisual_loading=True,
)

name = 'napkin'
yes = sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (0, 0),
    angle = 0,
    size = obj_dict.default_size[name],
    rgb= colors['sliver'],
    scale_factor= obj_dict.default_scale_factor[name],
    material = 'cloth',
    texture=True
)
print(yes)

name = 'cup1'
yes = sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (20, 20),
    angle = 0,
    size = obj_dict.default_size[name],
    rgb=colors['sliver'],
    scale_factor= obj_dict.default_scale_factor[name],
    material = None,
    texture=True
)
print(yes)
name = 'cup1'
yes = sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (20, 20),
    angle = 0,
    size = obj_dict.default_size[name],
    rgb=(140, 140, 140,),
    scale_factor= obj_dict.default_scale_factor[name],
    material = None,
    texture=True
)
print(yes)
name = 'cup1'
yes = sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (32, 32),
    size = obj_dict.default_size[name],
    angle = 0,
    rgb=(140, 140, 140,),
    scale_factor= obj_dict.default_scale_factor[name],
    material = 'metallic',
    texture=True
)
print(yes)
sim.get_observation_nvisii("./exps/1/")
sim.reset()

name = 'cup5'
sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (20, 20),
    angle = 0,
    size = obj_dict.default_size[name],
    rgb=colors['red'],
    scale_factor= obj_dict.default_scale_factor[name],
    material = 'metallic',
    texture=False
)
name = 'plate2'
sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (30, 20),
    size = obj_dict.default_size[name],
    angle = 0,
    rgb=None,
    scale_factor= obj_dict.default_scale_factor[name],
    material = None,
    texture=True
)
sim.get_observation_nvisii("./exps/2/")