from tabletop_gym.envs.pb_env_nvisii import Tabletop_Sim
import tabletop_gym.envs.object as obj_dict

color_labels = []
material_labels = []
type_labels = []

sim = Tabletop_Sim(
    width=640,
    height=640,
    indivisual_loading=True,
)

name = 'cup1'
yes = sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (0, 0),
    angle = 0,
    size = obj_dict.default_size[name],
    rgb=(140, 140, 140,),
    scale_factor= obj_dict.default_scale_factor[name],
    material = None,
    texture=True
)
print(yes)
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
yes = sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (32, 32),
    size = obj_dict.default_size[name],
    angle = 0,
    rgb=(140, 140, 140,),
    scale_factor= obj_dict.default_scale_factor[name],
    material = None,
    texture=True
)
print(yes)
sim.get_observation_nvisii("./exps/1/")
sim.reset()

sim.load_object(
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
sim.load_object(
    name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (32, 32),
    size = obj_dict.default_size[name],
    angle = 0,
    rgb=(140, 140, 140,),
    scale_factor= obj_dict.default_scale_factor[name],
    material = None,
    texture=True
)
sim.get_observation_nvisii("./exps/2/")