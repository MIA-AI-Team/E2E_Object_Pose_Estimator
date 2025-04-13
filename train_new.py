import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Or "Qt5Agg" if using PyQt
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader
import open3d as o3d

try:
    from p4_helper import *
except ImportError:
    print("Warning: Could not import p4_helper. Make sure it's in your PYTHONPATH.")

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

try:
    from pose_cnn import PoseCNN, FeatureExtraction, SegmentationBranch, TranslationBranch, RotationBranch
    
    feature_extractor = FeatureExtraction(pretrained_model=vgg16)
    segmentation_branch = SegmentationBranch()
    translation_branch = TranslationBranch()
    rotation_branch = RotationBranch()
except ImportError:
    print("Warning: Could not import pose_cnn modules. Make sure they're in your PYTHONPATH.")

plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["font.size"] = 16
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

import multiprocessing

NUM_CLASSES = 10
BATCH_SIZE = 1
NUM_WORKERS = multiprocessing.cpu_count() // 2  # Half the cores to avoid OOM
# NUM_WORKERS=0
path = os.getcwd()
PATH = os.path.join(path)

# Select device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available, using CPU")

# Update the import to use BPRPoseDataset
from utils.BPRPoseDataset import BPRPoseDataset
import utils

# Function to load point clouds from PLY files
def load_point_clouds(model_paths, num_points=1000):
    """!git clone https://github.com/MIA-AI-Team/E2E_Object_Pose_Estimator.git

    Load point clouds from model paths
    
    Args:
        model_paths: Dictionary mapping object IDs to file paths
        num_points: Number of points to sample from each model
        
    Returns:
        Dictionary mapping object IDs to point clouds
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Error: open3d is not installed. Please install it with 'pip install open3d'")
        return {}
    
    model_pcds = {}
    for obj_id, file_path in model_paths.items():
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Warning: Model file {file_path} does not exist")
                continue
                
            # Load the point cloud using Open3D
            pcd = o3d.io.read_point_cloud(file_path)
            
            # Check if point cloud is empty
            if len(pcd.points) == 0:
                print(f"Warning: Empty point cloud for object {obj_id}")
                continue
                
            # Sample or limit points
            points = np.asarray(pcd.points, dtype=np.float32)
            if len(points) > num_points:
                # Randomly sample points
                indices = np.random.choice(len(points), num_points, replace=False)
                points = points[indices]
            elif len(points) < num_points:
                # Repeat points to reach desired number
                indices = np.random.choice(len(points), num_points - len(points), replace=True)
                points = np.vstack([points, points[indices]])
                
            model_pcds[obj_id] = points
        except Exception as e:
            print(f"Error loading point cloud for object {obj_id}: {e}")
    
    return model_pcds

# Create a dummy model for missing point clouds
def create_dummy_point_cloud(num_points=500):
    """Create a dummy point cloud for objects without models"""
    # Create a cube point cloud
    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                points.append([x, y, z])
    
    # Repeat points to reach desired number
    points = np.array(points, dtype=np.float32)
    if num_points > 8:
        indices = np.random.choice(8, num_points - 8, replace=True)
        points = np.vstack([points, points[indices]])
    
    return points




def get_data():
    """Use BPRPoseDataset for training and validation."""
    try:
        # Define the root directory of the dataset
        root = "/home/seif-ai/pose_cnn_project/E2E_Object_Pose_Estimator/ipd"

        train_dataset = BPRPoseDataset(root=root, split="train")
        val_dataset = BPRPoseDataset(root=root, split="val")

        # Limit the training dataset to the first n samples for testing
        train_dataset.samples = train_dataset.samples[:2]

        return train_dataset, val_dataset

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None


def main():
    try:
        utils.reset_seed(0)
        
        train_dataset, val_dataset = get_data()
        
        # Create data loaders with error handling
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=True,
            pin_memory=True if DEVICE.type == 'cuda' else False,
            persistent_workers=True if NUM_WORKERS > 0 else False,
        )
        
        # Handle point cloud models
        print("Processing point cloud models...")
        if not train_dataset.models_pcd:
            print("Warning: No model point clouds found in dataset.")
        
        # Try loading point clouds
        model_pcds = load_point_clouds(train_dataset.models_pcd, num_points=500)
        dummy_pcd = create_dummy_point_cloud(num_points=500)
        
        # Create point cloud tensor list for each object ID in obj_id_list
        model_pcds_tensor = []
        for obj_id in train_dataset.obj_id_list:
            if obj_id in model_pcds:
                model_pcds_tensor.append(torch.tensor(model_pcds[obj_id], dtype=torch.float32))
            else:
                print(f"Warning: No point cloud for object ID {obj_id}, using dummy")
                model_pcds_tensor.append(torch.tensor(dummy_pcd, dtype=torch.float32))
        
        # Stack into a single tensor
        models_pcd_tensor = torch.stack(model_pcds_tensor).to(DEVICE)
        print(f"Created point cloud tensor with shape: {models_pcd_tensor.shape}")
        
        print("Initializing PoseCNN model...")
        # Replace default_cam_intrinsic with a dynamically loaded or explicitly defined intrinsic matrix
        cam_intrinsic = np.array([[3280.98092, 0.0, 1200.0],
                                   [0.0, 3280.98092, 1200.0],
                                   [0.0, 0.0, 1.0]])  # Example intrinsic matrix
        posecnn_model = PoseCNN(
            pretrained_backbone=vgg16,
            models_pcd=models_pcd_tensor,
            cam_intrinsic=cam_intrinsic  # Pass the intrinsic matrix explicitly
        ).to(DEVICE)
        posecnn_model.train()

        optimizer = torch.optim.Adam(posecnn_model.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        loss_history = []
        log_period = 5
        _iter = 0
        st_time = time.time()
        epochs = 20  
        
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            torch.cuda.empty_cache()  # Clear GPU cache

            train_loss = []
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    print(f"Batch {batch_idx} objs_id: {batch['objs_id']}")
                    print(f"train_dataset.obj_id_list: {train_dataset.obj_id_list}")
                    for item in batch:
                        batch[item] = batch[item].to(DEVICE)

                    if batch['objs_id'].max() >= len(train_dataset.obj_id_list):
                        print(f"Warning: Invalid object ID in batch {batch_idx}. Skipping.")
                        continue

                    loss_dict = posecnn_model(batch)

                    optimizer.zero_grad()
                    total_loss = sum(loss for loss in loss_dict.values())
                    total_loss.backward()
                    optimizer.step()

                    train_loss.append(total_loss.item())

                    if _iter % log_period == 0:
                        loss_str = f"[Epoch {epoch+1}/{epochs}][Batch {batch_idx+1}/{len(train_loader)}][Iter {_iter}][loss: {total_loss:.3f}]"
                        for key, value in loss_dict.items():
                            loss_str += f"[{key}: {value:.3f}]"
                        print(loss_str)
                        loss_history.append(total_loss.item())

                    _iter += 1
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    continue

            elapsed_time = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))
            if train_loss:
                avg_loss = np.mean(train_loss)
                print(f"Time {elapsed_time}, Epoch {epoch+1}/{epochs}, Training finished with mean loss {avg_loss:.3f}")
            else:
                print(f"Time {elapsed_time}, Epoch {epoch+1}/{epochs}, Training finished but no valid loss recorded")

        # Save model
        save_path = os.path.join(PATH, "posecnn_model.pth")
        print(f"Saving model to {save_path}")
        torch.save(posecnn_model.state_dict(), save_path)

        if loss_history:
            plt.figure()
            plt.title("Training loss history")
            plt.xlabel(f"Iteration (x {log_period})")
            plt.ylabel("Loss")
            plt.plot(loss_history)
            plt.savefig(os.path.join(PATH, "loss_history.png"))
        else:
            print("No loss history to plot")
            
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
