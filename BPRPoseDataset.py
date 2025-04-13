import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import open3d as o3d  # For handling .PLY models


class BPRPoseDataset(Dataset):
    def __init__(self, root: str, split: str = 'train_pbr', max_instance_num: int = 10) -> None:
        """
        Initialize the BPRPoseDataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split ('train' or 'val').
            max_instance_num (int): Maximum number of instances per image.
        """
        assert split in ['train_pbr', 'val'], "Split must be 'train_pbr' or 'val'."

        self.root = root
        self.split = split
        self.dataset_dir = os.path.join(self.root, split)
        self.max_instance_num = max_instance_num
        self.H, self.W = 480, 640  # Image dimensions

        # Parse dataset files
        self.rgb_dir = os.path.join(self.dataset_dir, "rgb_cam1")
        self.depth_dir = os.path.join(self.dataset_dir, "depth_cam1")
        self.mask_dir = os.path.join(self.dataset_dir, "mask_cam1")
        self.scene_gt_file = os.path.join(self.dataset_dir, "scene_gt_cam1.json")  # Updated file name
        self.scene_gt_info_file = os.path.join(self.dataset_dir, "scene_gt_info_cam1.json")  # Updated file name
        self.scene_camera_file = os.path.join(self.dataset_dir, "scene_camera_cam1.json")
        self.models_info_file = os.path.join(self.root, "models", "models_info.json")
        self.models_dir = os.path.join(self.root, "models")  # Corrected directory path

        # Load JSON files
        self.scene_gt = self._load_json(self.scene_gt_file)
        self.scene_gt_info = self._load_json(self.scene_gt_info_file)
        self.scene_camera = self._load_json(self.scene_camera_file)
        self.models_info = self._load_json(self.models_info_file)

        self.obj_id_list = sorted({obj['obj_id'] for frame in self.scene_gt.values() for obj in frame})
        self.id2label = {obj_id: idx for idx, obj_id in enumerate(self.obj_id_list)}  # Map obj_id to index        # Parse model point clouds
        self.models_pcd = self.parse_model()

        self.samples = self._prepare_samples()

    def _load_json(self, file_path: str):
        """Load a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        with open(file_path, 'r') as f:
            return json.load(f)

# filepath: /home/seif-ai/pose_cnn_project/E2E_Object_Pose_Estimator/utils/BPRPoseDataset.py
    def parse_model(self):
        """
        Parse the models directory to load point cloud models.
        """
        model_path = os.path.join(self.root, "models")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model directory {model_path} does not exist.")
            return {}
        
        obj_files = {}
        for file in os.listdir(model_path):
            if file.endswith('.ply'):
                try:
                    # Example: "obj_000001.ply" -> extract the id as an integer.
                    obj_id = int(file.split('_')[1].split('.')[0])
                    obj_files[obj_id] = os.path.join(model_path, file)
                except Exception as e:
                    print(f"Error parsing model file {file}: {e}")
        
        print(f"Found {len(obj_files)} model files: {list(obj_files.keys())}")
        return obj_files
    
    def _prepare_samples(self):
        """Prepare dataset samples."""
        samples = []
        for frame_id, objects in self.scene_gt.items():
            rgb_path = os.path.join(self.rgb_dir, f"{frame_id.zfill(6)}.jpg")
            depth_path = os.path.join(self.depth_dir, f"{frame_id.zfill(6)}.png")
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue
            samples.append((frame_id, rgb_path, depth_path))
        return samples

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        frame_id, rgb_path, depth_path = self.samples[idx]
        objs_dict = self.scene_gt[frame_id]
        objs_info_dict = self.scene_gt_info[frame_id]
        camera_info = self.scene_camera[frame_id]

        # Load RGB and depth images
        with Image.open(rgb_path) as im:
            rgb = np.array(im.resize((640, 480))).astype(np.float32) / 255.0  # Resize to 640x480
        with Image.open(depth_path) as im:
            depth = np.array(im.resize((640, 480))).astype(np.float32) * camera_info["depth_scale"]
        # Camera intrinsics and pose
        cam_K = np.array(camera_info["cam_K"]).reshape(3, 3)
        cam_R_w2c = np.array(camera_info["cam_R_w2c"]).reshape(3, 3)
        cam_t_w2c = np.array(camera_info["cam_t_w2c"]).reshape(3, 1)

        # Initialize data structures
        objs_id = np.zeros(self.max_instance_num, dtype=np.uint8)
        label = np.zeros((self.max_instance_num + 1, self.H, self.W), dtype=bool)
        bbx = np.zeros((self.max_instance_num, 4))
        RTs = np.zeros((self.max_instance_num, 3, 4))
        centers = np.zeros((self.max_instance_num, 2))

        # Process objects in the frame
        for obj_idx, obj_data in enumerate(objs_dict):
            if obj_idx >= self.max_instance_num:
                break
            obj_id = obj_data['obj_id']
            if obj_id in self.id2label:  # Ensure obj_id is valid
                objs_id[obj_idx] = self.id2label[obj_id]  # Map obj_id to index

                # Bounding box
                bbox = objs_info_dict[obj_idx]['bbox_obj']
                bbx[obj_idx] = bbox

                # Pose (R, T)
                R = np.array(obj_data['cam_R_m2c']).reshape(3, 3)
                T = np.array(obj_data['cam_t_m2c']).reshape(3, 1)
                RTs[obj_idx, :, :3] = R
                RTs[obj_idx, :, 3] = T.flatten()

                # Center
                center_homo = cam_K @ T
                centers[obj_idx] = center_homo[:2].flatten() / center_homo[2]

        data_dict = {
            'rgb': rgb.transpose((2, 0, 1)),  
            'depth': depth[np.newaxis, :],  # Add channel dimension
            'objs_id': objs_id,
            'label': label,
            'bbx': bbx,
            'RTs': RTs,
            'centers': centers,
            'cam_intrinsic': cam_K,
            'cam_R_w2c': cam_R_w2c,
            'cam_t_w2c': cam_t_w2c
        }

        return data_dict
    

dataset = BPRPoseDataset(root = "/home/seif-ai/pose_cnn_project/E2E_Object_Pose_Estimator/ipd"
, split="train_pbr")
data = dataset[0]

print("RGB Shape:", data['rgb'].shape)
print("Depth Shape:", data['depth'].shape)
print("Bounding Boxes:", data['bbx'])
print("RT Matrices:", data['RTs'])
print("Camera Intrinsics:", data['cam_intrinsic'])
print("Camera Rotation (w2c):", data['cam_R_w2c'])
print("Camera Translation (w2c):", data['cam_t_w2c'])
