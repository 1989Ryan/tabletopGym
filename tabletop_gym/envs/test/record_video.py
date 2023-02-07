import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from PIL import Image
import numpy as np
from expert.motion import pick_and_place_action

from pb_env_nvisii import Tabletop_Sim
from utils import coord_from_pixel, read_json
import nvisii
import subprocess
from paragon.paraground import models
from paragon.object_detector.run import maskrcnn_obj_detecotr
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from paragon.paraground.utils.argument import parse_args
import random

def read_record_file(path):
    '''
    read the file and return the ids for simulator loading 
    '''
    folders = os.listdir(path)
    data = []
    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        mode, num ,_, _ = folder.split("_")
        num = int(num)
        for file in files:
            filepath = os.path.join(path, folder, file)
            if '.json' in file:
                file = file.replace('.json', '')
                original_filepath = os.path.join("/home/zirui/tabletop_gym/dataset/", folder.replace('cliport', 'nvisii'), file, 'info_compositional.json')
                info_original = read_json(original_filepath)
                target_id = info_original['target_id']
                task_id, table_id, step_id = file.split('_')
                ids = [int(task_id), int(table_id), int(step_id), mode, num, int(target_id)]
                info = read_json(filepath)
                data.append([ids, info])
    return data

def read_dataset(folderpath, task_id, table_id, step_id, comp):
    if comp:
        info = read_json(f"{folderpath}/{task_id}_{table_id}_{step_id}/info_compositional.json")
    else:
        info = read_json(f"{folderpath}/{task_id}_{table_id}_{step_id}/info_simple.json")
    return info

def pick_object(names, id_names, get_object_position, obj_id):
    # print('object names are {}'.format(names))
    # print('please input the object id: ')
    # name_id = input()
    # name_json = read_json('/home/zirui/tabletop_gym/tabletop_gym/envs/config/id2labelBB.json')
    # name = name_json[str(obj_id)]
    
    # name_id = id_names(name)
    # pose = None
    # if 'wine' in name:
    #     pos, orn = get_object_position(int(name_id))
    #     pos1, pos2, pos3 = pos
    #     pos3 = pos3 + 0.13
    #     pos_new = (pos1, pos2, pos3)
    # elif '' in name:
    #     pos, orn = get_object_position(int(name_id))
    #     pos1, pos2, pos3 = pos
    #     pos3 = pos3 + 0.10
    #     pos_new = (pos1, pos2, pos3)
    pos, orn = get_object_position(int(obj_id + 3))
    pos1, pos2, pos3 = pos
    pos3 = pos3 + 0.02
    pos_new = (pos1, pos2, pos3)
    return pos_new

def compute_placement(samples, weights):
    weights = weights.exp().squeeze()
    pred = random.choices(samples.view(-1, 2), weights=weights, k=1)
    pred = pred[0]
    print(pred)
    # print(pred * 640)
    coord = coord_from_pixel(np.array([pred[1], pred[0]]) * 640)
    coord = coord.reshape(3)
    return coord.tolist()

def pick_and_place(pick_xyz, place_xyz, planner, sim):
    # sim.start_video()
    pos = pick_xyz
    _, _, height = pick_xyz
    place_1, place_2, place_3 = place_xyz[0]
    pose = (pos, None)
    action = planner.pre_post_pick_place_from_top(pose)
    sim.apply_action(action)
    ee_sensor = sim.ee.detect_contact()
    pos1, pos2, pos3 = pos
    pos = (pos1, pos2, pos3)
    pose = (pos, None)
    action=planner.pick_from_top(pose, ee_sensor)
    sim.apply_action(action, interp_n=500)
    ee_sensor = sim.ee.detect_contact()
    img, _, _, _ = sim.get_observation()
    while not ee_sensor:
        # pos, orn = sim.get_obj_pose(18)
        pos1, pos2, pos3 = pos
        pos = (pos1, pos2, pos3-0.05)
        pose = (pos, None)
        action=planner.pick_from_top(pose, ee_sensor)
        sim.apply_action(action, interp_n=150)
        ee_sensor = sim.ee.detect_contact()
        img, _, _, _ = sim.get_observation()

    pos1, pos2, pos3 = pos
    pos = (pos1, pos2, pos3-0.05)
    pose = (pos, None)
    action=planner.pick_from_top(pose, ee_sensor)
    sim.apply_action(action, interp_n=150)
    action=planner.pre_post_pick_place_from_top(pose)
    sim.apply_action(action)
    pos = (place_1, place_2, height)
    pose = (pos, None)
    action=planner.pre_post_pick_place_from_top(pose)
    sim.apply_action(action)
    ee_sensor = sim.ee.detect_contact()
    pos1, pos2, pos3 = pos
    pos = (pos1, pos2, pos3)
    pose = (pos, None)
    action = planner.place_from_top(pose, ee_sensor, pose)
    sim.apply_action(action, interp_n = 500)
    ee_sensor = sim.ee.detect_contact()
    while not ee_sensor:
        pos1, pos2, pos3 = pos
        pos = (pos1, pos2, pos3-0.05)
        pose = (pos, None)
        action = planner.place_from_top(pose, ee_sensor, pose)
        sim.apply_action(action, interp_n = 150)
        ee_sensor = sim.ee.detect_contact()

    action=planner.place_from_top(pose, ee_sensor, pose)
    sim.apply_action(action)
    action=planner.pre_post_pick_place_from_top(pose)
    sim.apply_action(action)
    sim.close_video()

def visualize_(coord, weights, img):
    # Create figure and axes
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img)
    plt.savefig("visualize1.png",  bbox_inches='tight')
    # Create a Rectangle patch
    # for idx in idxs:
    #     bbox = bboxes[idx]
    #     # plt.text(bbox[0], bbox[3], str(v[order[counter]].item()))
    #     rect = patches.Rectangle((bbox[0], bbox[3]), bbox[2]-bbox[0], bbox[1]-bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    #     # Add the patch to the Axes
    #     ax.add_patch(rect)
    #     counter += 1
    # rect = patches.Circle((placement[0].cpu().detach().numpy(), placement[1].cpu().detach().numpy()), 40, linewidth=1, edgecolor='b', facecolor='none')
    # Add the patch to the Axes
    # ax.add_patch(rect)
    # coord = coord.reshape(-1, 2)
    coord = coord.squeeze()
    if weights is not None:
        ax.scatter((coord[:,0] * 640).cpu().detach().numpy(), 
            (coord[:,1]*640).cpu().detach().numpy(), label='target',s=2000*(weights.exp().cpu().detach().numpy()), c='#76EE00')
    else:
        ax.scatter((coord[0] * 640).cpu().detach().numpy(), 
            (coord[1]*640).cpu().detach().numpy(), label='target',s=50, c='#76EE00')
 
    # plt.show()
    plt.savefig("visualize.png",  bbox_inches='tight' )

def main(args):

    data_num =      args.data_num
    attn =          args.attn
    load_model =    args.load_model
    parser =        'attn' if args.attn else 'lin'
    attribute =     'rs_rp_rnn'
    if not args.use_rs_rp:
        if args.use_rp_only:
            if args.no_rw:
                attribute = 'no_rs_no_rw'
            else:
                attribute = 'no_rs'
        elif args.use_rs_only:
            attribute = 'no_rp'
        else:
            attribute = 'no_rs_rp'
    elif not args.soft_parsing:
        attribute = 'no_parser'
    elif not args.is_rand:
        attribute = 'no_particle_rnn'
    elif args.single:
        attribute = 'rs_rp_rnn_single'
    task = 'simple relations'    
    if args.eval_comp:
        task = 'compositional relations'
    print("######### evaluation information ############")
    print('testing model: {}'.format('{}_{}'.format(args.lang_model, attribute)))
    print('testing file: {}'.format(args.model_file))
    print('######### model details ##########')
    print('particle number: {}'.format(args.particle_num))
    print('resampling alpha: {}'.format(args.resamp_alpha))
    print('MFGNN layers: {}'.format(args.layer_num))
    print('feature embedding dim: {}'.format(args.embd_dim))
    print("######### evaluation process ###########")
    device = torch.device(args.dev_name) if torch.cuda.is_available() else torch.device('cpu')
    model = models['{}_{}'.format(args.lang_model, attribute)](
        aggr=args.aggr,
        word_embd_dim=args.word_embd_dim,
        embd_dim=args.embd_dim,
        gnn_layer_num=args.layer_num,
        particle_num=args.particle_num,
        resamp_alpha=args.resamp_alpha,
        position_size=args.position_size,
        device=device
    )
    model.to(device)
    model.eval()
    f1 = torch.load(args.model_file, map_location=device)
    model.load_state_dict(f1)
    # dataset = tabletop_gym_obj_dataset('/home/zirui/tabletop_gym/dataset/test_4_obj_nvisii', test=True)
    # data_n = len(dataset)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=1, shuffle=True, num_workers=0)
    obj_detect = maskrcnn_obj_detecotr('/home/zirui/paraground/trained_model/0_9_mask_rcnn.pt')



    sim = Tabletop_Sim(
        from_bullet=False, 
        state_id=388, 
        table_id = 508, 
        step_id=7, 
        mode='train',
        width=640,
        height=640,
        use_gui=False,
        obj_number=11,
        record_video=False,
        record_cfg = None,
    )
    planner = pick_and_place_action()
    
    state_id, table_id, step_id, mode, num, target_id = 1, 306, 5, 'test', 11, 11
    info = read_dataset('/home/zirui/paraground/dataset/test_11_obj_nvisii', state_id, table_id, step_id, comp=True)
    record_cfg = {
        'save_video_path': '/home/zirui/paraground/record_video_simple/',
        'video_name': f'{mode}_{num}_{state_id}_{table_id}_{step_id}_1.mp4',
        'gui_name': f'{mode}_{num}_{state_id}_{table_id}_{step_id}_2.mp4',
        'fps': 30,
    }
    sim.reset_with_param(state_id=state_id, table_id=table_id, step_id=step_id, mode = mode,\
                    save_imgs=True, from_bullet=False, obj_number=num, record_video=True, record_cfg=record_cfg, 
                    obj_list=list(info['state'].keys()))
    # weights = np.array(info['weights'])
    # samples = np.array(info['pred'])
    # place_xyz = compute_placement(samples, weights)
    sim.get_observation_nvisii('test_observation/')
    raw_img = Image.open(
        'test_observation/rgb.png'
    ).convert("RGB")
    print(sim.obj_names)
    img = F.pil_to_tensor(raw_img)
    img = F.convert_image_dtype(img).unsqueeze(0)
    bboxes, scores = obj_detect.query(img)
    while True:
        print("type in your instruction:")
        ins = input()
        if args.is_rand:
            if args.single:
                pred_, tar_w, coord_tensor, p_h = model([ins], [bboxes], img)
                pred, weights = pred_
            elif args.no_rw:
                pred_, tar_prob, bbox_coord = model(ins, [bboxes], img)
                pred = pred_
                pred_size = pred.size()
                weights = torch.log(torch.ones(size=(pred_size[0], pred_size[1], pred_size[2], 1), device=device)/pred_size[2])
            else:
                pred_, tar_prob, bbox_coord = model(ins, [bboxes], img)
                pred, weights = pred_
        else:
            pred, tar_w, coord_tensor = model(ins, [bboxes], img)

        visualize_(pred[:, -1], weights[:, -1], raw_img)
        print("execute? (Y/n)")
        yes = input()
        if yes == 'y':
            print(sim.object_property)
            print("target_id: ")
            target_id = input()
            target_id = int(target_id)
            break
    place_xyz = compute_placement(pred[:, -1], weights[:, -1])
    # place_xyz = coord_from_pixel(np.array([info['goal_pixel'][0], info['goal_pixel'][1]]))
    pick_xyz = pick_object(sim.obj_names, sim.name2id, sim.get_obj_pose, target_id)
    pick_and_place(pick_xyz, place_xyz, planner, sim)
        # print('press any key to continue: ')
        # input()
    nvisii.deinitialize()
    subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"rgb_%d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath('./video'))
    subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"rgb_gui_%d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output_gui.mp4'], cwd=os.path.realpath('./video'))

if __name__ == '__main__':
    args = parse_args()
    main(args)