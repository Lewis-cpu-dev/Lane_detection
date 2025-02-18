import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.lane_detector import LaneDetector
from models.enet import ENet
from datasets.lane_dataset import LaneDataset
from torch.utils.data import DataLoader
import os

# Define dataset and checkpoint paths
DATASET_PATH = "/opt/data/TUSimple"
# DATASET_PATH =  "/home/lzy/.cache/kagglehub/datasets/manideep1108/tusimple/versions/5/TUSimple/test_set"
CHECKPOINT_PATH = "checkpoints/BalanceLoss/enet_checkpoint_epoch_96.pth"  # Path to the trained model checkpoint
BATCH_SIZE = 8
# Function to load the ENet model
def load_enet_model(checkpoint_path, device="cuda"):
    enet_model = ENet(binary_seg=2, embedding_dim=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    enet_model.load_state_dict(checkpoint['model_state_dict'])
    enet_model.eval()
    return enet_model

def calculate_cross(pt1, pt2, bottom=720):
    rowBottom = bottom
    
    row1, col1 = pt1
    row2, col2 = pt2
    
    slope = (col1 - col2)/(row1 - row2)
    
    colBottom = col1 - (row1 - rowBottom) * slope
    
    return [round(rowBottom), round(colBottom)]

def perspective_transform(image):
    """
    Transform an image into a bird's eye view.
        1. Calculate the image height and width.
        2. Define source points on the original image and corresponding destination points.
        3. Compute the perspective transform matrix using cv2.getPerspectiveTransform.
        4. Warp the original image using cv2.warpPerspective to get the transformed output.
    """
    
    ####################### TODO: Your code starts Here #######################
    
    # ptBL = calculate_cross([320,480],[500,0],720)  # [720, -587]
    # ptBR = calculate_cross([320,799],[500,1279],720)  # [720, 1866]
    
    # image = cv2.resize(imageOuter, (1280, 720)).astype(np.float32)
    # print(image.shape)  # (720, 1280, 3)
    height, width = image.shape[:2]  # print(image.shape)  # (720, 1280, 3)
    
    # ptTL = [320,480] ; ptBL = [719, -584]
    # ptTR = [320,799] ; ptBR = [719, 1863]
    ptTL = [113,191] ; ptBL = [255, -233]
    ptTR = [113,319] ; ptBR = [255, 744]
    srcPts = np.float32([ptTL[::-1], ptBL[::-1], ptTR[::-1], ptBR[::-1]])
    # srcPts[:, 0] = srcPts[:, 0]/1279*511
    # srcPts[:, 1] = srcPts[:, 1]/719*255
    # print(srcPts)
    
    
    imgTL = [0,0] ; imgBL = [255, 0]
    imgTR = [0,511] ; imgBR = [255, 511]
    dstPts = np.float32([imgTL[::-1], imgBL[::-1], imgTR[::-1], imgBR[::-1]])
    # print(dstPts)
    
    # cv2.imshow("image", image)
    TransformMatrix = cv2.getPerspectiveTransform(srcPts, dstPts)
    # cv2.findHomography(srcPts, dstPts, solveMethod=cv2.RANSAC)  # cv2.RANSAC, cv2.LMEDS, cv2.RHO, 0
    transformed_image = cv2.warpPerspective(image, TransformMatrix, (width, height))
    
    ####################### TODO: Your code ends Here #######################
    return transformed_image

def mapColor(origin,img):
    color_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    color_img[np.where(img == 0)] = [0, 0, 0]    # 0 -> 黑色
    color_img[np.where(img == 1)] = [255, 0, 0]  # 1 -> 红色
    color_img[np.where(img == 2)] = [0, 255, 0]  # 2 -> 绿色
    color_img[np.where(img == 3)] = [0, 0, 255]  # 3 -> 蓝色
    color_img[np.where(img == -1)] = [0, 0, 255]  # 3 -> 蓝色
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = (img/img.max())*255
    img = img.astype(np.uint8)
    lane_mask = img>30
    lane_colored = cv2.applyColorMap(img,cv2.COLORMAP_VIRIDIS)
    lane_colored = cv2.cvtColor(lane_colored, cv2.COLOR_BGR2RGB)
    alpha = 0.7
    overlay = origin.copy()
    purp = np.full_like(overlay,(128,0,128),dtype=np.uint8)
    overlay = cv2.addWeighted(overlay,0.8,purp,0.2,0)
    overlay[lane_mask] = cv2.addWeighted(img,1-alpha,lane_colored,alpha,0)[lane_mask]
    
    
    
    
    return color_img,overlay
    

# Function to visualize lane predictions for multiple images in a single row
def visualize_lanes_row(images, instances_maps,labels,alpha=0.7):
    """
    Visualize lane predictions for multiple images in a single row
    For each image:
        1. Resize it to 512 x 256 for consistent visualization.
        2. Apply perspective transform to both the original image and its instance map.
        3. Overlay the instance map to a plot with the corresponding original image using a specified alpha value.
    """
    
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    # ####################### TODO: Your code starts Here #######################
    for i in range(num_images):
        print(i)
        image = cv2.resize(images[i], dsize=(512, 256))
        instances_map = np.array(instances_maps[i]) # * 255 / np.max(instances_maps[i])
        instances_color_map,colored_lane = mapColor(image,instances_map)
        transformed_img = perspective_transform(image).astype(np.uint8)
        transformed_color_map = perspective_transform(instances_color_map).astype(np.uint8)
        
        rbg_instance = instances_maps[i]
        instance_emb = rbg_instance[:,:,:3]  # Instance embedding tensor
        minv = np.min(instance_emb)
        maxv = np.max(instance_emb)
        instance_vis = 255*(instance_emb-minv)/(maxv-minv+1e-6)
        instance_vis_rgb =  instance_vis.astype(np.uint8)
        
        cv2.imshow("image", image)
        cv2.imshow("instances_map", colored_lane)
        cv2.imshow("Ground Truth segmentation", labels[i])
        cv2.imshow("transformed_img", transformed_img)
        cv2.imshow("transformed_map", transformed_color_map)
        cv2.imshow("Embbeded", instance_vis_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        overlay = cv2.addWeighted(transformed_img, alpha, transformed_color_map, 1 - alpha, 0).astype(np.uint8)
        axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))  # OpenCV 是 BGR，要转成 RGB
        axes[i].axis("off")

    # ####################### TODO: Your code ends Here #######################



def main():
    # Initialize device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enet_model = load_enet_model(CHECKPOINT_PATH, device)
    lane_predictor = LaneDetector(enet_model, device=device)
    
    val_dataset = LaneDataset(dataset_path=DATASET_PATH, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # List of test image paths
    sub_paths = [
        "train_set/clips/0313-1/60/20.jpg",
        "train_set/clips/0313-1/120/20.jpg",
        "train_set/clips/0313-2/20/20.jpg",
        "train_set/clips/0313-2/70/20.jpg",
        "train_set/clips/0601/1494452433570893263/20.jpg"
    ]
    test_image_paths = [os.path.join(DATASET_PATH, sub_path) for sub_path in sub_paths]
    
    label_subpaths = [
    	"train_set/seg_label/0313-1/60/20.png",
        "train_set/seg_label/0313-1/120/20.png",
        "train_set/seg_label/0313-2/20/20.png",
        "train_set/seg_label/0313-2/70/20.png",
        "train_set/seg_label/0601/1494452433570893263/20.png"
    ]
    test_label_paths = [os.path.join(DATASET_PATH, label_subpath) for label_subpath in label_subpaths]
    threshold = 1
    label_images = []
    for label in test_label_paths:
    	limage = cv2.imread(label,cv2.IMREAD_GRAYSCALE)
    	_, biimage = cv2.threshold(limage,threshold,255,cv2.THRESH_BINARY)
    	label_images.append(biimage)
    # Load and process images
    images = []
    instances_maps = []
    
    for path in test_image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Unable to load image at {path}")
            continue

        print(f"Processing image: {path}")
        instances_map = lane_predictor(image)
        images.append(image)
        instances_maps.append(instances_map)

    # Visualize all lane predictions in a single row
    if images and instances_maps:
        visualize_lanes_row(images, instances_maps,label_images)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

