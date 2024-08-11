import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.model_selection import train_test_split
import numpy as np
import pybullet as p
import pybullet_data
from concurrent.futures import ThreadPoolExecutor
import cv2
import math
import random

# LET'S SET SEED FOR REPRODUCIBILITY
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# LET'S DEFINE THE STEREO MODEL CLASS
class StereoNet(torch.nn.Module):
    def __init__(self, in_channels):
        super(StereoNet, self).__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        self.refiner = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 1, kernel_size=3, padding=1),  # OUTPUTTING 1 CHANNEL FOR DEPTH MAP
            torch.nn.Upsample(size=(480, 640), mode='bilinear', align_corners=True)  # ADJUSTING THE SIZE TO 640x480
        )

    def forward(self, left_images, right_images):
        left_features = self.feature_extractor(left_images)
        right_features = self.feature_extractor(right_images)
        combined_features = torch.cat((left_features, right_features), dim=1)
        depth_map = self.refiner(combined_features)
        return depth_map

# WE DEFINE THE PyBulletStereoDataset class
class PyBulletStereoDataset(Dataset):
    def __init__(self, root_dir):
        self.left_dir = os.path.join(root_dir, 'left')
        self.right_dir = os.path.join(root_dir, 'right')
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.left_images = sorted(os.listdir(self.left_dir))
        self.right_images = sorted(os.listdir(self.right_dir))
        self.depth_maps = sorted(os.listdir(self.depth_dir))

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img_path = os.path.join(self.left_dir, self.left_images[idx])
        right_img_path = os.path.join(self.right_dir, self.right_images[idx])
        depth_map_path = os.path.join(self.depth_dir, self.depth_maps[idx])

        left_img, right_img, depth_map = None, None, None
        retry_count = 3
        while retry_count > 0:
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)
            depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

            if left_img is not None and right_img is not None and depth_map is not None:
                break
            retry_count -= 1

        if left_img is None or right_img is None or depth_map is None:
            return None

        left_img = torch.tensor(left_img).permute(2, 0, 1).float() / 255.0
        right_img = torch.tensor(right_img).permute(2, 0, 1).float() / 255.0
        depth_map = torch.tensor(depth_map).unsqueeze(0).float() / 255.0  # ADDING CHANNEL DIMENSION

        return left_img, right_img, depth_map

# WE ADD A FUNCTION TO CREATE AND SAVE THE DATASET USING PyBullet
def move_drones_and_capture_depth(save_data_dir):
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    os.makedirs(os.path.join(save_data_dir, 'left'), exist_ok=True)
    os.makedirs(os.path.join(save_data_dir, 'right'), exist_ok=True)
    os.makedirs(os.path.join(save_data_dir, 'depth'), exist_ok=True)
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    environment_urdf = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\SketchFab_Blender_NYC Environment\enviro.urdf"
    environment_orientation = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
    p.loadURDF(environment_urdf, basePosition=[0, 0, 9], baseOrientation=environment_orientation, useFixedBase=True, globalScaling=7)

    drone_urdf = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 1\rotors_simulator\rotors_description\urdf\ardrone.urdf"
    drone1 = p.loadURDF(drone_urdf, [1, -1, 1])
    drone2 = p.loadURDF(drone_urdf, [1, -1, 1])
    drone3 = p.loadURDF(drone_urdf, [1, 1, 1])

    initial_orientation_drone3 = p.getQuaternionFromEuler([0, 0, -np.deg2rad(0)])

    combinations = [
        (90, 0.1, 50),
        (90, 0.2, 100),
        (90, 0.3, 150),
        (120, 0.1, 50),
        (120, 0.2, 100),
        (120, 0.3, 150),
        (150, 0.1, 50),
        (150, 0.2, 100),
        (150, 0.3, 150)
    ]

    def capture_and_save_images(drone_id, idx, fov, near, far):
        width, height = 640, 480  # WE ADJUST THE RESOLUTION TO 640x480
        aspect = width / height
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        
        left_camera_trans = [0.2, 0.1, 0.1]
        right_camera_trans = [0.2, -0.1, 0.1]

        left_camera = p.multiplyTransforms(p.getBasePositionAndOrientation(drone_id)[0], p.getBasePositionAndOrientation(drone_id)[1], left_camera_trans, [0, 0, 0, 1])
        right_camera = p.multiplyTransforms(p.getBasePositionAndOrientation(drone_id)[0], p.getBasePositionAndOrientation(drone_id)[1], right_camera_trans, [0, 0, 0, 1])

        left_view_matrix = p.computeViewMatrix(
            cameraEyePosition=left_camera[0],
            cameraTargetPosition=[left_camera[0][0] + 1, left_camera[0][1], left_camera[0][2]],
            cameraUpVector=[0, 0, 1]
        )
        
        right_view_matrix = p.computeViewMatrix(
            cameraEyePosition=right_camera[0],
            cameraTargetPosition=[right_camera[0][0] + 1, right_camera[0][1], right_camera[0][2]],
            cameraUpVector=[0, 0, 1]
        )

        left_img = p.getCameraImage(width, height, viewMatrix=left_view_matrix, projectionMatrix=proj_matrix)[2]
        right_img = p.getCameraImage(width, height, viewMatrix=right_view_matrix, projectionMatrix=proj_matrix)[2]
        depth_img = p.getCameraImage(width, height, viewMatrix=left_view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_TINY_RENDERER)[3]

        left_img = np.reshape(left_img, (height, width, 4))[:, :, :3].astype(np.uint8)
        right_img = np.reshape(right_img, (height, width, 4))[:, :, :3].astype(np.uint8)
        
        depth_buffer = np.reshape(depth_img, (height, width))
        depth_img = far * near / (far - (far - near) * depth_buffer)

        depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_img_rgb = cv2.applyColorMap(depth_img_normalized, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(save_data_dir, 'left', f'left_{idx:04d}_fov{fov}_near{near}_far{far}.png'), left_img)
        cv2.imwrite(os.path.join(save_data_dir, 'right', f'right_{idx:04d}_fov{fov}_near{near}_far{far}.png'), right_img)
        cv2.imwrite(os.path.join(save_data_dir, 'depth', f'depth_{idx:04d}_fov{fov}_near{near}_far{far}.png'), depth_img_rgb)

    def circular_motion(t, radius=5):
        x = radius * math.cos(t)
        y = radius * math.sin(t)
        return [x, y, 14]

    def zigzag_motion(t, amplitude=2, frequency=0.5):
        x = amplitude * math.sin(frequency * t)
        y = amplitude * math.cos(frequency * t)
        return [x, y, 14]

    def random_motion(t, amplitude=2):
        x = amplitude * math.sin(t)
        y = amplitude * math.cos(t) * math.sin(t)
        return [x, y, 14]

    motion_patterns = [circular_motion, zigzag_motion, random_motion]

    executor = ThreadPoolExecutor(max_workers=3)
    idx = 0
    for fov, near, far in combinations:
        for step in range(10):  
            t = step * 0.1
            drone1_position = circular_motion(t)
            drone2_position = zigzag_motion(t)
            drone3_position = random_motion(t)

            p.resetBasePositionAndOrientation(drone1, drone1_position, [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(drone2, drone2_position, [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(drone3, drone3_position, initial_orientation_drone3)

            futures = [executor.submit(capture_and_save_images, drone1, idx, fov, near, far),
                       executor.submit(capture_and_save_images, drone2, idx, fov, near, far),
                       executor.submit(capture_and_save_images, drone3, idx, fov, near, far)]
            for future in futures:
                future.result()
            idx += 1

    p.disconnect()

# WRAPPER THAT WILL HANDLE THE LIST OF SAMPLES AS A DATASET
class DatasetWrapper(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        return self.dataset_list[idx]

# THE FOLLOWING IS A FUNCTION TO TRAIN STEREONET MODEL WITH LR SCHEDULING AND VALIDATION SET
def train_stereonet_with_scheduler(model, train_dataset, val_dataset, epochs, batch_size, learning_rate):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            left_images, right_images, gt_depth_maps = data
            
            optimizer.zero_grad()
            
            outputs = model(left_images, right_images)
            
            loss = criterion(outputs, gt_depth_maps)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                left_images_val, right_images_val, gt_depth_maps_val = val_data
                val_outputs = model(left_images_val, right_images_val)
                val_loss += criterion(val_outputs, gt_depth_maps_val).item()
        
        print(f'Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')
        
        scheduler.step(val_loss)
        
        evaluate_model(model, val_dataset)

# THE FOLLOWING IS A FUNCTION TO EVALUATE STEREONET MODEL WITH SSIM AND PSNR METRICS (SSIM: STRUCTURAL SIMILARITY INDEX & PSNR: PEAK SIGNAL-TO-NOISE RATIO)
def evaluate_model(model, dataset):
    model.eval()
    ssim_scores = []
    psnr_scores = []
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for data in dataloader:
            left_img, right_img, gt_depth = data
            
            outputs = model(left_img, right_img)
            
            pred_depth = outputs.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            
            ssim_score = ssim(pred_depth, gt_depth, data_range=gt_depth.max() - gt_depth.min())
            psnr_score = psnr(pred_depth, gt_depth, data_range=gt_depth.max() - gt_depth.min())
            
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
    
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    
    print(f'Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}')

# THE FOLLOWING IS THE MAIN FUNCTION
if __name__ == "__main__":
    save_data_dir = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\TRAINING STEREONET & LSTM BEST RESULTS\training_data"
    move_drones_and_capture_depth(save_data_dir)

    dataset = PyBulletStereoDataset(root_dir=save_data_dir)
    dataset_list = [sample for sample in [dataset[i] for i in range(len(dataset))] if sample is not None]
    
    train_dataset, val_dataset = train_test_split(dataset_list, test_size=0.2, random_state=SEED)
    train_dataset = DatasetWrapper(train_dataset)
    val_dataset = DatasetWrapper(val_dataset)

    model = StereoNet(in_channels=3)

    train_stereonet_with_scheduler(model, train_dataset, val_dataset, epochs=20, batch_size=2, learning_rate=0.001)

    save_path = r'C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\TRAINING STEREONET & LSTM BEST RESULTS\trained_stereonet.pth'
    torch.save(model.state_dict(), save_path)

    print(f"Trained model saved to {save_path}")
