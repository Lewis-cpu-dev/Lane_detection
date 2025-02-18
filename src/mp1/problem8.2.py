import torch
import os
import numpy as np
import wandb
from torch.utils.data import DataLoader
from datasets.lane_dataset import LaneDataset
from models.enet import ENet
from models.losses import compute_loss
from utils.visualization import visualize_first_prediction,visualize_listed_prediction
from utils.lane_detector import LaneDetector
import matplotlib.pyplot as plt
import cv2

# Configuration
BATCH_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH =  "/opt/data/TUSimple"
CHECKPOINT_PATH = "checkpoints/BalanceLoss/enet_checkpoint_epoch_96.pth"  # Path to the trained model checkpoint


def evaluate():
    """
    Evaluate the trained ENet model on a validation dataset and log results to Weights and Biases.
    """

    val_dataset = LaneDataset(dataset_path=DATASET_PATH, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    enet_model = ENet(binary_seg=2, embedding_dim=4).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    enet_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint successfully loaded from {CHECKPOINT_PATH} (Epoch {checkpoint['epoch']})")

    enet_model.eval()
    binary_losses, instance_losses, total_losses = [], [], []
    print(f"length of val{len(val_loader)} \n and {val_loader}")
    with torch.no_grad():
        im = 0
        for images, binary_labels, instance_labels in val_loader:
            
            print(images.shape)
            print(im)
            im+=1
            images = images.to(DEVICE)
            binary_labels = binary_labels.to(DEVICE)
            instance_labels = instance_labels.to(DEVICE)

            binary_logits, instance_embeddings = enet_model(images)

            binary_loss, instance_loss = compute_loss(
                binary_output=binary_logits,
                instance_output=instance_embeddings,
                binary_label=binary_labels,
                instance_label=instance_labels,
            )
            total_loss = binary_loss + instance_loss

            binary_losses.append(binary_loss.item())
            instance_losses.append(instance_loss.item())
            total_losses.append(total_loss.item())
            
            # Visualize the first prediction in the batch
            combined_rows = visualize_listed_prediction([0,10,20,40],
                images.cpu(),
                binary_logits.cpu(),
                instance_embeddings.cpu(),
                binary_labels.cpu(),
                instance_labels.cpu()
            )

            # Log the first combined row to W&B
            for combined_row in combined_rows:
            	cv2.imshow("eval",combined_row)
            	cv2.waitKey(0)
            break  # Process only the first batch for visualization
#    print(f"length: {len(instance_losses[0].shape)}")
    mean_binary_loss = np.mean(binary_losses)
    mean_instance_loss = np.mean(instance_losses)
    mean_total_loss = np.mean(total_losses)

    

if __name__ == '__main__':
    for i in range(5):
      evaluate()
