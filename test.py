from tabletop_gym.envs.pb_env_nvisii import Tabletop_Sim
import tabletop_gym.envs.object as obj_dict
import os

episode = 0
base_path = f"./exps/{episode}"
os.makedirs(base_path, exist_ok = True)
obs_counter = 0

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
    "042_adjustable_wrench",
    "070-b_colored_wood_blocks",
    "031_spoon",
    "038_padlock",
    "055_baseball",
    "029_plate",
    "030_fork",
    "073-e_lego_duplo",
    "013_apple",
    "058_golf_ball",
    "032_knife",
    "022_windex_bottle",
    "008_pudding_box",
    "002_master_chef_can",
    "061_foam_brick",
    "007_tuna_fish_can",
    "073-c_lego_duplo",
    "065-f_cups",
    "063-a_marbles",
    "065-h_cups",
    "019_pitcher_base",
    "006_mustard_bottle",
    "073-b_lego_duplo",
    "073-f_lego_duplo",
    "050_medium_clamp",
    "073-d_lego_duplo",
    "056_tennis_ball",
    "062_dice",
    "017_orange",
    "077_rubiks_cube",
    "010_potted_meat_can",
    "057_racquetball",
    "014_lemon",
    "026_sponge",
    "009_gelatin_box",
    "065-a_cups",
    "051_large_clamp",
    "063-b_marbles",
    "053_mini_soccer_ball",
    "024_bowl",
    "016_pear",
    "065-b_cups",
    "018_plum",
    "035_power_drill",
    "012_strawberry",
    "037_scissors",
    "025_mug",
    "072-d_toy_airplane",
    "015_peach",
    "059_chain",
    "065-e_cups",
    "065-i_cups",
    "065-c_cups",
    "073-g_lego_duplo",
    "004_sugar_box",
    "052_extra_large_clamp",
    "072-b_toy_airplane",
    "021_bleach_cleanser",
    "011_banana",
    "071_nine_hole_peg_test",
    "065-j_cups",
    "065-g_cups",
    "072-e_toy_airplane",
    "073-a_lego_duplo",
    "072-a_toy_airplane",
    "044_flat_screwdriver",
    "070-a_colored_wood_blocks",
    "043_phillips_screwdriver",
    "054_softball",
    "072-c_toy_airplane",
    "028_skillet_lid",
    "033_spatula",
    "036_wood_block",
    "003_cracker_box",
    "048_hammer",
    "065-d_cups",
    "040_large_marker",
    "005_tomato_soup_can"
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

sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
obs_counter += 1

# name = 'napkin'

# yes = sim.load_object(
#     type_name = obj_dict.filename[name],
#     mesh_name = obj_dict.type[name],
#     baseMass = 1,
#     position = (0, 0),
#     angle = 0,
#     size = obj_dict.default_size[name],
#     rgb= colors['sliver'],
#     scale_factor= obj_dict.default_scale_factor[name],
#     material = 'cloth',
#     texture=True
# )
# # output whether there is no collision. if False, the object
# # will not be loaded
# print(yes) 
# sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
# obs_counter += 1



name = '011_banana'
object_name = "mycup"
sim.load_object(
    name = object_name,
    type_name = obj_dict.filename[name],
    mesh_name = obj_dict.type[name],
    baseMass = 1,
    position = (0, 20),
    angle = 0,
    size = obj_dict.default_size[name],
    rgb=colors['sliver'],
    scale_factor= obj_dict.default_scale_factor[name],
    material = None,
    texture=True
)
sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
obs_counter += 1

sim.reset_obj_pose(name=object_name, 
                size=obj_dict.default_size[name], 
                position=(10, 10), 
                baseOrientationAngle=0)
sim._dummy_run()
sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")

# name = 'cup1'
# yes = sim.load_object(
#     type_name = obj_dict.filename[name],
#     mesh_name = obj_dict.type[name],
#     baseMass = 1,
#     position = (20, 20),
#     angle = 0,
#     size = obj_dict.default_size[name],
#     rgb=(140, 140, 140,),
#     scale_factor= obj_dict.default_scale_factor[name],
#     material = None,
#     texture=True
# )
# print(yes)
# sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
# obs_counter += 1

# name = 'cup1'
# yes = sim.load_object(
#     type_name = obj_dict.filename[name],
#     mesh_name = obj_dict.type[name],
#     baseMass = 1,
#     position = (32, 32),
#     size = obj_dict.default_size[name],
#     angle = 0,
#     rgb=(140, 140, 140,),
#     scale_factor= obj_dict.default_scale_factor[name],
#     material = 'metallic',
#     texture=True
# )
# print(yes)
# # sim.get_observation_nvisii("./exps/1.png")
# sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
# obs_counter += 1
# # sim.reset()

# name = 'cup5'
# sim.load_object(
#     type_name = obj_dict.filename[name],
#     mesh_name = obj_dict.type[name],
#     baseMass = 1,
#     position = (20, 20),
#     angle = 0,
#     size = obj_dict.default_size[name],
#     rgb=colors['red'],
#     scale_factor= obj_dict.default_scale_factor[name],
#     material = 'metallic',
#     texture=False
# )
# sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
# obs_counter += 1

# name = 'plate2'
# sim.load_object(
#     type_name = obj_dict.filename[name],
#     mesh_name = obj_dict.type[name],
#     baseMass = 1,
#     position = (30, 20),
#     size = obj_dict.default_size[name],
#     angle = 0,
#     rgb=None,
#     scale_factor= obj_dict.default_scale_factor[name],
#     material = None,
#     texture=True
# )
# sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
# obs_counter += 1
# # sim.get_observation_nvisii("./exps/2.png")
