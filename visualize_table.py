from tabletop_gym.envs.pb_env_nvisii import Tabletop_Sim
import tabletop_gym.envs.object as obj_dict
import os
import table_config.object_spec as object_spec
import pdb
import numpy as np
import json
import random


def get_single_tidy_config(episode="single"):

    base_path = f"./exps/{episode}"
    os.makedirs(base_path, exist_ok = True)
    obs_counter = 0

    object_list, grid = object_spec.get_tidy_config()
    np.savetxt(f'{base_path}/grid.out', grid)
                        
    pdb.set_trace()
    num_objs = len(object_list)

    sim = Tabletop_Sim(
        width=640,
        height=640,
        indivisual_loading=True,
    )

    for i in range(num_objs):

        obj = object_list[i]
        obj_name = obj["name"]
        obj_type = obj["type"]

        if obj["color"] is None:
            pdb.set_trace()

        yes = sim.load_object(
            name = obj_name,
            type_name = obj_dict.filename[obj_type],
            mesh_name = obj_dict.type[obj_type],
            baseMass = 1,
            position = obj["pos"],
            angle = obj["angle"],
            size = obj_dict.default_size[obj_type],
            rgb= obj["color"],
            scale_factor= obj_dict.default_scale_factor[obj_type],
            material = None,
            texture=True
        )
        
        if yes is False:
            print(f"Collison caused by obj {name}!")
            pdb.set_trace()
        else:
            sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
            obs_counter += 1

def batch_tidy_config(episode="batch_init", render=True):
  
    base_path = f"./exps/{episode}"
    os.makedirs(base_path, exist_ok = True)

    batch_tidy_config = object_spec.batch_initialize_tidy_config()
    with open(f'{base_path}/config.json', 'w') as fout:
        json.dump(batch_tidy_config, fout)
                        
    num_configs = len(batch_tidy_config)
    print(f"{num_configs} initial tidy configurations" )

    if render is False:
        return 

    sim = Tabletop_Sim(
            width=640,
            height=640,
            indivisual_loading=True,
        )

    for conf in batch_tidy_config:
        
        object_list, grid, spec = conf

        sim = init_config(sim, object_list)
        sim.get_observation_nvisii(f"{base_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_hori_{spec[3]}_left_{spec[4]}.png")

def batch_random_walk(episode="batch_traj"):
    root_path = f"./exps/{episode}"
    os.makedirs(root_path, exist_ok = True)

    with open(f'./exps/batch_init/config.json', 'r') as f:
        batch_tidy_config = json.load(f)
                        
    pdb.set_trace()
    num_configs = len(batch_tidy_config)
    print(f"{num_configs} initial tidy configurations" )

    sim = Tabletop_Sim(
            width=640,
            height=640,
            indivisual_loading=True,
    )

    for conf in batch_tidy_config:
        object_list, grid, spec = conf
        grid = np.array(grid)
        base_path = f"{root_path}/cup1_{spec[0]}_cup2_{spec[1]}_fork_{spec[2]}_hori_{spec[3]}_left_{spec[4]}"
        os.makedirs(base_path, exist_ok = True)

        # initialize the tidy configuration
        sim = init_config(sim, object_list)
        sim.get_observation_nvisii(f"{base_path}/0.png")

        random.shuffle(object_list)

        for t in range(len(object_list)):
            obj = object_list[t] 
            obj, grid = object_spec.random_walk_one_step(obj, grid)
            object_list[t] = obj
            sim.reset_obj_pose(name=obj["name"], 
                size=obj_dict.default_size[obj["type"]], 
                position=obj["pos"], 
                baseOrientationAngle=0)
        
            sim.get_observation_nvisii(f"{base_path}/{t+1}.png")
        # pdb.set_trace()

               
def init_config(sim, object_list):

    sim.reset()
    num_objs = len(object_list)
    for i in range(num_objs):

        obj = object_list[i]
        obj_name = obj["name"]
        obj_type = obj["type"]

        yes = sim.load_object(
                name = obj_name,
                type_name = obj_dict.filename[obj_type],
                mesh_name = obj_dict.type[obj_type],
                baseMass = 1,
                position = obj["pos"],
                angle = obj["angle"],
                size = obj_dict.default_size[obj_type],
                rgb= obj["color"],
                scale_factor= obj_dict.default_scale_factor[obj_type],
                material = None,
                texture=True
        )
        # sim.get_observation_nvisii(f"{base_path}/{obs_counter}.png")
        # obs_counter += 1
        if yes is False:
            print(f"Collison!")
            pdb.set_trace()
    return sim


batch_tidy_config(render=False)
batch_random_walk()