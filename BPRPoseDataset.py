# import os
# import numpy as np
# import cv2
# from PIL import Image
# from torch.utils.data import Dataset

# import json
# from typing import Dict, Any

# class BPRPoseDataset(Dataset):
#     def __init__(self, root: str, rgb_file: str, depth_file: str, mask_file: str, cam_K: list, depth_scale: float) -> None:
#         """
#         Initialize the dataset for a single image.

#         Args:
#             root (str): Root directory of the dataset.
#             rgb_file (str): Path to the RGB image.
#             depth_file (str): Path to the depth image.
#             mask_file (str): Path to the mask file.
#             cam_K (list): Camera intrinsic parameters.
#             depth_scale (float): Depth scale factor.
#         """
#         self.root = root
#         self.rgb_file = os.path.join(root, rgb_file)
#         self.depth_file = os.path.join(root, depth_file)
#         self.mask_file = os.path.join(root, mask_file)
        
#         # Get the scene_gt_info file path
#         scene_id = "cam1"  # Changed from parsing the path
#         frame_id = os.path.basename(rgb_file).split('.')[0]  # e.g., "000000"
#         self.gt_info_file = os.path.join(root, "train", f"scene_gt_info_{scene_id}.json")
#         self.frame_id = frame_id
        
#         self.cam_K = np.array(cam_K).reshape(3, 3)
#         self.depth_scale = depth_scale

#         # Parameters
#         self.H = 480
#         self.W = 640
#         self.max_instance_num = 10

#         # Default camera intrinsic parameters
#         self.default_cam_intrinsic = np.array([
#             [902.19, 0.0, 342.35],
#             [0.0, 902.39, 252.23],
#             [0.0, 0.0, 1.0]
#         ])

#         # Parse models
#         self.models_pcd = self.parse_model()

#         # Initialize object ID list
#         self.obj_id_list = [0, 1, 4, 8, 10, 11, 14, 18, 19, 20]
#         self.id2label = {obj_id: idx + 1 for idx, obj_id in enumerate(self.obj_id_list)}
        
#         # Load ground truth info
#         self.gt_info = self.load_gt_info()

#     def load_gt_info(self) -> Dict[str, Any]:
#         """Load ground truth info from JSON file."""
#         try:
#             if os.path.exists(self.gt_info_file):
#                 with open(self.gt_info_file, 'r') as f:
#                     gt_info = json.load(f)
#                     return gt_info.get(self.frame_id, {})
#             else:
#                 print(f"Warning: GT info file not found: {self.gt_info_file}")
#                 return {}
#         except Exception as e:
#             print(f"Error loading GT info: {e}")
#             return {}

#     def get_bbox_from_gt(self, obj_idx: int) -> np.ndarray:
#         """Get bounding box from ground truth info with validation."""
#         try:
#             if self.gt_info and str(obj_idx) in self.gt_info:
#                 bbox = self.gt_info[str(obj_idx)]['bbox_obj']
#                 # Validate bbox values
#                 x, y, w, h = bbox
#                 if w <= 0 or h <= 0:
#                     return self.get_default_bbox()
                
#                 # Ensure bbox stays within image bounds
#                 x1 = max(0, x)
#                 y1 = max(0, y)
#                 x2 = min(self.W, x + w)
#                 y2 = min(self.H, y + h)
                
#                 return np.array([x1, y1, x2, y2])
#             return self.get_default_bbox()
#         except Exception as e:
#             print(f"Error getting bbox: {e}")
#             return self.get_default_bbox()
        
#     def get_default_bbox(self) -> np.ndarray:
#         """Return a reasonable default bounding box."""
#         # Use 80% of image size as default bbox
#         margin = 0.1
#         w, h = self.W, self.H
#         x1 = int(w * margin)
#         y1 = int(h * margin)
#         x2 = int(w * (1 - margin))
#         y2 = int(h * (1 - margin))
#         return np.array([x1, y1, x2, y2])    

#     def __getitem__(self, idx):
#         try:
#             # Load and resize the RGB image
#             with Image.open(self.rgb_file) as im:
#                 rgb = np.array(im)
#                 rgb = cv2.resize(rgb, (self.W, self.H))
#             rgb = rgb.astype(np.float32) / 255
#             rgb = rgb.transpose((2, 0, 1))  # Convert to CHW format

#             # Load and resize the depth image
#             with Image.open(self.depth_file) as im:
#                 depth = np.array(im)
#                 depth = cv2.resize(depth, (self.W, self.H))
#             depth = (depth * self.depth_scale)[np.newaxis, :]

#             # Load and resize the mask
#             with Image.open(self.mask_file) as im:
#                 mask = np.array(im, dtype=bool)
#                 mask = cv2.resize(mask.astype(np.uint8), (self.W, self.H), 
#                                 interpolation=cv2.INTER_NEAREST).astype(bool)

#             # Create object data
#             objs_id = np.zeros(self.max_instance_num, dtype=np.uint8)
#             label = np.zeros((self.max_instance_num + 1, self.H, self.W), dtype=bool)
#             bbx = np.zeros((self.max_instance_num, 4))
#             RTs = np.zeros((self.max_instance_num, 3, 4))
#             centers = np.zeros((self.max_instance_num, 2))
#             centermaps = np.zeros((self.max_instance_num, 3, self.H, self.W))

#             # Populate object data
#             # objs_id[0] = self.id2label[1]  # Map object ID 1 to its label
#             # label[objs_id[0]] = mask
#             # label[0] = ~mask  # Background is the inverse of the object mask
            
#             # # Get bbox from ground truth info
#             # bbx[0] = self.get_bbox_from_gt(1)  # Using object ID 1
            
#             # RTs[0] = np.zeros((3, 4))
#             # centers[0] = [self.W // 2, self.H // 2]
#             # centermaps[0] = np.zeros((3, self.H, self.W))
#             obj_id = 1  # Using object ID 1 as example
#             objs_id[0] = self.id2label[obj_id]
#             label[objs_id[0]] = mask
#             label[0] = ~mask  # Background is the inverse of the object mask
            
#             # Get bbox and ensure it's valid
#             bbox = self.get_bbox_from_gt(obj_id)
#             bbx[0] = bbox
            
#             # Calculate center from bbox
#             center_x = (bbox[0] + bbox[2]) / 2
#             center_y = (bbox[1] + bbox[3]) / 2
#             centers[0] = [center_x, center_y]
            
#             # Create simple centermap (Gaussian blob around center)
#             x = np.arange(0, self.W)
#             y = np.arange(0, self.H)
#             X, Y = np.meshgrid(x, y)
#             sigma = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 6  # Scale with bbox size
#             centermaps[0, 0] = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))

            
#             data_dict = {
#                 'rgb': rgb,
#                 'depth': depth,
#                 'objs_id': objs_id,
#                 'label': label,
#                 'bbx': bbx,
#                 'RTs': RTs,
#                 'centermaps': centermaps.reshape(-1, self.H, self.W),
#                 'centers': centers,
#                 'cam_intrinsic': self.cam_K,
#                 'depth_scale': self.depth_scale
#             }

#             return data_dict

#         except Exception as e:
#             print(f"Error processing sample: {e}")
#             return self._get_dummy_sample()

#     def parse_model(self):
#         """
#         Parse the models_eval directory to load point cloud models.
#         """
#         model_path = os.path.join(self.root, "models_eval")
        
#         # Check if model directory exists
#         if not os.path.exists(model_path):
#             print(f"Warning: Model directory {model_path} does not exist.")
#             return {}
        
#         obj_files = {}
#         for file in os.listdir(model_path):
#             if file.endswith('.ply'):
#                 try:
#                     # Example: "obj_000001.ply" -> extract the id as an integer.
#                     obj_id = int(file.split('_')[1].split('.')[0])
#                     obj_files[obj_id] = os.path.join(model_path, file)
#                 except Exception as e:
#                     print(f"Error parsing model file {file}: {e}")
        
#         # Print found models
#         print(f"Found {len(obj_files)} model files: {list(obj_files.keys())}")
#         return obj_files




#     def __len__(self) -> int:
#         """Return the length of the dataset (1 for single image)"""
#         return 1

import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import open3d as o3d  # For handling .PLY models


class BPRPoseDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', max_instance_num: int = 10) -> None:
        """
        Initialize the BPRPoseDataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split ('train' or 'val').
            max_instance_num (int): Maximum number of instances per image.
        """
        assert split in ['train', 'val'], "Split must be 'train' or 'val'."

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
        self.models_info_file = os.path.join(self.root, "models_eval", "models_info.json")
        self.models_dir = os.path.join(self.root, "models_eval")  # Corrected directory path

        # Load JSON files
        self.scene_gt = self._load_json(self.scene_gt_file)
        self.scene_gt_info = self._load_json(self.scene_gt_info_file)
        self.scene_camera = self._load_json(self.scene_camera_file)
        self.models_info = self._load_json(self.models_info_file)

        self.obj_id_list = sorted({obj['obj_id'] for frame in self.scene_gt.values() for obj in frame})
        self.id2label = {obj_id: idx for idx, obj_id in enumerate(self.obj_id_list)}  # Map obj_id to index        # Parse model point clouds
        self.models_pcd = self.parse_model()

        # Prepare dataset samples
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
        Parse the models_eval directory to load point cloud models.
        """
        model_path = os.path.join(self.root, "models_eval")
        
        # Check if model directory exists
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
        
        # Print found models
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
            rgb = np.array(im).astype(np.float32) / 255.0
        with Image.open(depth_path) as im:
            depth = np.array(im).astype(np.float32) * camera_info["depth_scale"]

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

        # Create data dictionary
        data_dict = {
            'rgb': rgb.transpose((2, 0, 1)),  # CHW format
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
    

dataset = BPRPoseDataset(root = "/home/seif-ai/pose_cnn_project/E2E_Object_Pose_Estimator/BPR-Dataset"
, split="train")
data = dataset[0]

print("RGB Shape:", data['rgb'].shape)
print("Depth Shape:", data['depth'].shape)
print("Bounding Boxes:", data['bbx'])
print("RT Matrices:", data['RTs'])
print("Camera Intrinsics:", data['cam_intrinsic'])
print("Camera Rotation (w2c):", data['cam_R_w2c'])
print("Camera Translation (w2c):", data['cam_t_w2c'])