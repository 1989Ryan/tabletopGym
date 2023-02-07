import os.path as osp
import sys

from matplotlib.pyplot import draw
from tabletop_gym.envs.utils import draw_boxes, dump_json, parse_csv_filename, get_json_path_from_scene,\
    read_json, pixel_from_coord, inv_transform, get_bbox2d_from_segmentation, \
    get_segmentation_mask_object_and_link_index
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from PIL import Image
import numpy as np
from expert.motion import pick_and_place_action

from pybullet_env import Tabletop_Sim

# sim = Tabletop_Sim()
sim = Tabletop_Sim(
    from_bullet=True, 
    state_id=388, 
    table_id = 508, 
    step_id=7, 
    mode='train',
    width=640,
    height=640,
    use_gui=False,
)

img, depth, mask, ins, _ = sim.get_observation_with_mask()
map1, map2 = get_segmentation_mask_object_and_link_index(mask)
obsmask = Image.fromarray(map1.astype(np.uint8))
map1 = np.array(obsmask)
obj_ids = np.unique(map1)
print(obj_ids)
obj_ids = obj_ids[2:]
masks = mask == obj_ids[:, None, None]
print(masks)
num_objs = len(obj_ids)
print(num_objs)
boxes = []
masks_idx = []
for i in range(num_objs):
    pos = np.where(masks[i])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    if xmin >= xmax or ymin >= ymax:
        num_objs -= 1
        masks_idx.append(False)
        continue
    else:
        boxes.append([xmin, ymin, xmax, ymax])
        masks_idx.append(True)
label = np.zeros((num_objs, ), dtype=np.int64)
counter = 0
for i in range(num_objs):
    pos = np.where(masks[i])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    if xmin >= xmax or ymin >= ymax:
        continue
    else:
        label[counter] = obj_ids[i] - obj_ids[0] + 1
        counter += 1

print(np.shape(masks[masks_idx]))
image = draw_boxes(img, np.array(boxes))
print(label)
id2label = {}
label2id = {}
for ele in label:
    print(ele)
    print(sim.id2name(ele + obj_ids[0] - 1).replace(" BB", ''))
    id2label[int(ele)] = sim.id2name(ele + obj_ids[0] - 1).replace(" BB", '')
    label2id[sim.id2name(ele + obj_ids[0] - 1).replace(" BB", '')] = int(ele)
dump_json(id2label, "id2label.json")
dump_json(label2id, "label2id.json")
im = Image.fromarray(image)
im.save('demo.png')