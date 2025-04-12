import json
import os
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from utils import Visualize, chromatic_transform, add_noise


class BOPDataset(Dataset):
    base_folder = "BOP-Dataset"

    def __init__(
            self,
            root: str,
            split: str = 'train',
            download: bool = False,
    ) -> None:
        assert split in ['train', 'val']

        self.root = root
        self.split = split
        self.dataset_dir = os.path.join(self.root, self.base_folder)

        if download:
            self.download()

        ## parameter
        self.max_instance_num = 5 # 10 for the whole dataset
        self.H = 2400
        self.W = 2400
        self.rgb_aug_prob = 0.4
        self.cam_intrinsic = np.array([
            [902.19, 0.0, 342.35],
            [0.0, 902.39, 252.23],
            [0.0, 0.0, 1.0]]) # needs to be checked
        self.resolution = [2400, 2400]

        self.all_lst = self.parse_dir()
        self.shuffle()
        self.models_pcd = self.parse_model()

        self.obj_id_list = [
            0,
            8,
            18,
            19,
            20
        ]
        self.id2label = {}
        for idx, id in enumerate(self.obj_id_list):
            self.id2label[id] = idx + 1

    def parse_dir(self):
        data_dir = os.path.join(self.dataset_dir, self.split)
        rgb_path = os.path.join(data_dir, "rgb_cam1-20250317T130825Z-001")
        depth_path = os.path.join(data_dir, "depth_cam1-20250317T130809Z-001")
        mask_path = os.path.join(data_dir, "mask_visib_cam1-20250317T130821Z-001")
        scene_gt_json = "BOP-Dataset/train/scene_gt_cam1.json"
        scene_gt_info_json = "BOP-Dataset/train/scene_gt_info_cam1.json"
        scene_camera_json = "BOP-Dataset/train/scene_camera_cam1.json"
        rgb_list = os.listdir(rgb_path)
        rgb_list.sort()
        depth_list = os.listdir(depth_path)
        depth_list.sort()
        mask_list = os.listdir(mask_path)
        mask_list.sort()
        scene_gt = json.load(open(scene_gt_json))
        scene_gt_info = json.load(open(scene_gt_info_json))
        assert len(rgb_list) == len(depth_list) == len(scene_gt) == len(scene_gt_info), "data files number mismatching"
        all_lst = []
        for rgb_file in rgb_list:
            idx = int(rgb_file.split(".jpg")[0])
            depth_file = f"{idx:06d}.png"
            scene_objs_gt = scene_gt[str(idx)]
            scene_objs_info_gt = scene_gt_info[str(idx)]
            objs_dict = {}
            for obj_idx in range(len(scene_objs_gt)):
                dic = {}
                try:
                    dic['R'] = np.array(scene_objs_gt[obj_idx]['cam_R_m2c']).reshape(3, 3)
                    dic['T'] = np.array(scene_objs_gt[obj_idx]['cam_t_m2c']).reshape(3, 1)
                    dic['obj_id'] = scene_objs_gt[obj_idx]['obj_id']
                    dic['bbox_visib'] = scene_objs_info_gt[obj_idx]['bbox_visib']
                    assert f"{idx:006d}_{obj_idx:06d}.png" in mask_list
                    dic['visible_mask_path'] = os.path.join(mask_path, f"{idx:006d}_{obj_idx:06d}.png")
                except IndexError:
                    print(f"{rgb_file}: {obj_idx}")
                    continue
                objs_dict[obj_idx] = dic
            """
            obj_sample = (rgb_path, depth_path, objs_dict)
            objs_dict = {
                0: {
                    cam_R_m2c:
                    cam_t_m2c:
                    obj_id:
                    bbox_visib:
                    visiable_mask_path:
                }
                ...
            }
            """
            obj_sample = (
                os.path.join(rgb_path, rgb_file),
                os.path.join(depth_path, depth_file),
                objs_dict
            )
            all_lst.append(obj_sample)
        return all_lst

    def parse_model(self):
        model_path = os.path.join(self.dataset_dir, "model")
        objpathdict = {
            0: ["obj_000000", "BOP-Dataset/model/obj_000000.ply"],
            8: ["obj_000008", "BOP-Dataset/model/obj_000008.ply"],
            18: ["obj_000018", "BOP-Dataset/model/obj_000018.ply"],
            19: ["obj_000019", "BOP-Dataset/model/obj_000019.ply"],
            20: ["obj_000020", "BOP-Dataset/model/obj_000020.ply"],
        }
        self.visualizer = Visualize(objpathdict, self.cam_intrinsic, self.resolution)
        models_pcd_dict = {index: np.array(self.visualizer.objnode[index]['mesh'].vertices) for index in
                           self.visualizer.objnode}
        models_pcd = np.zeros((len(models_pcd_dict), 1024, 3))
        i=0
        for m in models_pcd_dict:
            model = models_pcd_dict[m]
            models_pcd[i] = model[np.random.randint(0, model.shape[0], 1024)]
            i+=1
        return models_pcd

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        """
        obj_sample = (rgb_path, depth_path, objs_dict)
        objs_dict = {
            0: {
                cam_R_m2c:
                cam_t_m2c:
                obj_id:
                bbox_visib:
                visiable_mask_path:
            }
            ...
        }

        data_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs',
            'centermaps', []
        }
        """
        rgb_path, depth_path, objs_dict = self.all_lst[idx]
        data_dict = {}
        with Image.open(rgb_path) as im:
            rgb = np.array(im)

        if self.split == 'train' and np.random.rand(1) > 1 - self.rgb_aug_prob:
            rgb = chromatic_transform(rgb)
            rgb = add_noise(rgb)
        rgb = rgb.astype(np.float32) / 255
        data_dict['rgb'] = rgb.transpose((2, 0, 1))

        with Image.open(depth_path) as im:
            data_dict['depth'] = np.array(im)[np.newaxis, :]
        objs_id = np.zeros(self.max_instance_num, dtype=np.uint8)
        label = np.zeros((self.max_instance_num + 1, self.H, self.W), dtype=bool)
        bbx = np.zeros((self.max_instance_num, 4))
        RTs = np.zeros((self.max_instance_num, 3, 4))
        centers = np.zeros((self.max_instance_num, 2))
        centermaps = np.zeros((self.max_instance_num, 3, self.resolution[1], self.resolution[0]))
        ## test
        img = cv2.imread(rgb_path)

        # Handle case when there are more objects than max_instance_num
        if len(objs_dict) > self.max_instance_num:
            # Sort by visible area if available, otherwise just take the first max_instance_num
            valid_objs = []
            for idx, obj in objs_dict.items():
                if len(obj['bbox_visib']) > 0:
                    area = obj['bbox_visib'][2] * obj['bbox_visib'][3]  # width * height
                    valid_objs.append((idx, area))

            # Sort by area in descending order and take top self.max_instance_num objects
            valid_objs.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [item[0] for item in valid_objs[:self.max_instance_num]]

            # Filter objs_dict to keep only selected objects
            filtered_objs_dict = {idx: objs_dict[idx] for idx in selected_indices}
            objs_dict = filtered_objs_dict

        # Process each object
        idx_counter = 0
        for orig_idx in objs_dict.keys():
            if len(objs_dict[orig_idx]['bbox_visib']) > 0:
                # Use a sequential index for the arrays, not the original object index
                objs_id[idx_counter] = self.id2label[objs_dict[orig_idx]['obj_id']]
                assert (objs_id[idx_counter] > 0)

                with Image.open(objs_dict[orig_idx]['visible_mask_path']) as im:
                    label[objs_id[idx_counter]] = np.array(im, dtype=bool)

                bbx[idx_counter] = objs_dict[orig_idx]['bbox_visib']

                RT = np.zeros((4, 4))
                RT[3, 3] = 1
                RT[:3, :3] = objs_dict[orig_idx]['R']
                RT[:3, [3]] = objs_dict[orig_idx]['T']
                RT = np.linalg.inv(RT)
                RTs[idx_counter] = RT[:3]

                center_homo = self.cam_intrinsic @ RT[:3, [3]]
                center = center_homo[:2] / center_homo[2]
                x = np.linspace(0, self.resolution[0] - 1, self.resolution[0])
                y = np.linspace(0, self.resolution[1] - 1, self.resolution[1])
                xv, yv = np.meshgrid(x, y)
                dx, dy = center[0] - xv, center[1] - yv
                distance = np.sqrt(dx ** 2 + dy ** 2)
                nx, ny = dx / distance, dy / distance
                Tz = np.ones((self.resolution[1], self.resolution[0])) * RT[2, 3]
                centermaps[idx_counter] = np.array([nx, ny, Tz])

                img = cv2.circle(img, (int(center[0]), int(center[1])), radius=2, color=(0, 0, 255), thickness=-1)
                centers[idx_counter] = np.array([int(center[0]), int(center[1])])

                idx_counter += 1

        label[0] = 1 - label[1:].sum(axis=0)

        data_dict['objs_id'] = objs_id
        data_dict['label'] = label
        data_dict['bbx'] = bbx
        data_dict['RTs'] = RTs
        data_dict['centermaps'] = centermaps.reshape(-1, self.resolution[1], self.resolution[0])
        data_dict['centers'] = centers

        return data_dict

    def shuffle(self):
        random.shuffle(self.all_lst)

    def download(self) -> None:
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
