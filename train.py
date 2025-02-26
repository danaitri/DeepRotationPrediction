#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for training a ResNet3D model on COCO Dataset for the task of angle prediction.

Contact Information:
    Name: Danai Triantafyllidou
    Email: danaitri22@gmail.com
    Date: 25-February-2025
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import argparse
import os
import random
import numpy as np
import wandb
from torchvision.utils import make_grid
from utils import rotate_preserve_size
import csv

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class COCODataset(Dataset):
    """
    Custom Dataset class for the COCO Dataset. There is the option to generate random rotation angles online (on-the-fly);
    This can be done by setting the parameter is_train=True. There is also the option to use a fixed set of angle parameters
    which can be obtained through a csv file; the latter is useful for logging consistent validation/test results.

    Args:
        root (str): Root directory containing the image dataset.
        is_train (bool, optional): If True we generate random rotation angles on-the-fly.
        angles_file (str, optional): Path to the CSV file containing angles for validation/test images. Defaults is None.

    Methods:
        load_angles_from_csv(angles_file): Loads rotation angles from a CSV file.
        __getitem__(index): Retrieves an image and its rotated version along with the rotation angle.
        __len__(): Returns the total number of images in the dataset.
    """

    def __init__(self, root, is_train=True, angles_file=None):
        self.root = root

        self.image_files = sorted(os.listdir(root))
        self.image_files = [x for x in self.image_files if x != ".DS_Store"]
        self.is_train = is_train

        self.transform = transform
        if not self.is_train and angles_file is not None:
            self.angles = self.load_angles_from_csv(angles_file)
        else:
            self.angles = None

    def load_angles_from_csv(self, angles_file):
        angles_dict = {}
        with open(angles_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                angles_dict[row['image_file']] = float(row['angle'])
        return [angles_dict[img_file] for img_file in self.image_files]

    def __getitem__(self, index):
        path = self.image_files[index]
        if os.path.exists(os.path.join(self.root, path)):
            img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.is_train:
            degree = random.uniform(0, 359)  # Generate a random rotation degree
        else:
            degree = self.angles[index]

        img_np = np.asarray(img)
        rotated_img_pil = rotate_preserve_size(img_np, degree, ((224, 224)))

        img_tensor = self.transform(img)
        rotated_tensor = self.transform(rotated_img_pil)

        norm_degree = degree
        return img_tensor, rotated_tensor, torch.tensor(norm_degree, dtype=torch.float32), self.image_files[index]

    def __len__(self):
        return len(self.image_files)


def calculate_total_params(model: nn.Module) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # Using a readable format with commas
    return total_params


class ResNet3D(nn.Module):
    """
    3D ResNet model for processing image pairs.
    """

    def __init__(self, num_classes: int = 359):
        super(ResNet3D, self).__init__()
        self.resnet3d = models.video.r3d_18(pretrained=True)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            input1 (torch.Tensor): The first input image tensor.
            input2 (torch.Tensor): The second input image tensor.

        Returns:
            torch.Tensor: Model output
        """
        input1 = input1.unsqueeze(2)  # Add a depth dimension to input1
        input2 = input2.unsqueeze(2)  # Add a depth dimension to input2
        x = torch.cat((input1, input2), dim=2)  # Concatenate along the depth dimension
        output = self.resnet3d(x)
        return output


def train(model: nn.Module, epochs: int = 10, save_frequency: int = 1, print_freq: int = 100, batch_size: int = 32,
          root_dir: str = "./", num_workers: int = 1,
          wandb_log: bool = True):
    """
    Train a ResNet3D model on the COCO Dataset.

    Args:
        model (nn.Module): The model to train.
        epochs (int, optional): Number of epochs for training.
        save_frequency (int, optional): Frequency of saving the model weights.
        print_freq (int, optional): Frequency of printing details about training.
        batch_size (int, optional): Batch size for model training.
        root_dir (str, optional): Root directory for the project.
        num_workers (int, optional): Number of workers for data loading.
        wandb_log (bool, optional): Enable or disable Weights & Biases logging.

    Example usage:
        train(model, epochs=10, save_frequency=1, print_freq=100, batch_size=32, root_dir="./", num_workers=1, wandb_log=True)

    Notes:
        This function logs the training and validation progress, saves model checkpoints, and logs images and metrics using Weights & Biases (if enabled).

    """

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Paths to training and validation datasets
    train_root = os.path.join(root_dir, 'data/coco2017/train2017/')
    val_root = os.path.join(root_dir, 'data/coco2017/val2017/')
    angles_file = os.path.join(root_dir, 'data/coco2017/val2017_angles.csv')

    # Create datasets
    train_dataset = COCODataset(root=train_root)
    val_dataset = COCODataset(root=val_root, is_train=False, angles_file=angles_file)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize the device and move the model to the device
    # r2_metric = R2Score().to(device)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0

        model.train()
        for i, (input1, input2, rotation_angle, _) in enumerate(train_loader):

            input1 = input1.to(torch.float32).to(device, non_blocking=True)
            input2 = input2.to(torch.float32).to(device, non_blocking=True)

            rotation_angle = rotation_angle.type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            output = model(input1, input2)

            loss = F.cross_entropy(output, rotation_angle, reduction="mean")
            predictions = torch.argmax(output, dim=1)
            correct_predictions = (torch.sum((predictions == rotation_angle))).float()
            accuracy = correct_predictions / batch_size

            absolute_errors = torch.abs(predictions - rotation_angle)
            mae = torch.mean(absolute_errors.float())
            running_mae += mae

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Log the loss to wandb
            if i % print_freq == 0:  # Print loss every x batches
                if wandb_log:
                    wandb.log({
                        "Batch Loss": loss.item(),
                        "Batch MAE": mae,
                    })
                print(
                    f'Epoch {epoch + 1}, Batch {i + 1}, Batch Loss: {loss.item():.6f}, Batch Accuracy: {accuracy:.6f}, Batch MAE: {mae:.6f}')

        train_loss = running_loss / len(train_loader)
        train_mae = running_mae / len(train_loader)

        model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for i, (input1, input2, rotation_angle, _) in enumerate(val_loader):

                input1 = input1.to(torch.float32).to(device, non_blocking=True)
                input2 = input2.to(torch.float32).to(device, non_blocking=True)

                rotation_angle = rotation_angle.type(torch.LongTensor).to(device)

                output = model(input1, input2)
                running_loss += F.cross_entropy(output, rotation_angle, reduction="mean")

                predictions = torch.argmax(output, dim=1)
                all_predictions.append(predictions)
                all_targets.append(rotation_angle)

                # Update R2 metric with predictions and targets
                # r2_metric.update(predictions, rotation_angle)

                # Log images and predictions
                if i % print_freq == 0:
                    if wandb_log:
                        images = []
                        for j in range(input1.size(0)):
                            img = wandb.Image(
                                make_grid([input1[j].cpu(), input2[j].cpu()], nrow=2),
                                caption=f"Pred: {predictions[j].item()  :.2f}°, Actual: {rotation_angle[j].item() :.2f}°"
                            )
                            images.append(img)
                        wandb.log({"Validation Images": images})

        val_loss = running_loss / len(val_loader)

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions).cpu()
        all_targets = torch.cat(all_targets).cpu()

        # Compute metrics
        val_mae = torch.mean(torch.abs(all_predictions.float() - all_targets.float())).item()
        # val_r2 = r2_metric.compute().item()

        print(f'Epoch {epoch + 1}')
        print(f'Training Loss: {train_loss:.4f}, Training MAE: {train_mae:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

        if wandb_log:
            wandb.log({
                "Train Loss": train_loss,
                "Train MAE": train_mae,
                "Val Loss": val_loss,
                "Val MAE": val_mae,
            })

        # Save the model checkpoint
        if (epoch + 1) % save_frequency == 0:
            model_save_path = root_dir + f'/siamese_network_checkpoint_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model checkpoint saved to {model_save_path}")


def test(model: nn.Module, print_freq: int = 100, batch_size: int = 32, root_dir: str = "./", num_workers: int = 1,
         wandb_log: bool = True) -> None:
    """
    Evaluate a Siamese ResNet model on the COCO Dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        print_freq (int, optional): Frequency of printing details about evaluation.
        batch_size (int, optional): Batch size for model testing.
        root_dir (str, optional): Root directory for the project.
        num_workers (int, optional): Number of workers for data loading.
        wandb_log (bool, optional): Enable or disable Weights & Biases logging.

    Example usage:
        test(model, print_freq=100, batch_size=32, root_dir="./", num_workers=1, wandb_log=True)

    Notes:
        This function evaluates the model on the test dataset, logs the results, and computes the evaluation metrics.
    """

    test_root = os.path.join(root_dir, 'data/coco2017/test2017/')
    angles_file = os.path.join(root_dir, 'data/coco2017/test2017_angles.csv')

    # Create datasets
    test_dataset = COCODataset(root=test_root, is_train=False, angles_file=angles_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Initialize the device and move the model to the device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    # r2_metric = R2Score().to(device)

    with torch.no_grad():
        for i, (input1, input2, rotation_angle, _) in enumerate(tqdm(test_loader, desc="Progress")):
            input1 = input1.to(torch.float32).to(device, non_blocking=True)
            input2 = input2.to(torch.float32).to(device, non_blocking=True)

            rotation_angle = rotation_angle.type(torch.LongTensor).to(device)

            output = model(input1, input2)
            running_loss += F.cross_entropy(output, rotation_angle, reduction="mean")

            predictions = torch.argmax(output, dim=1)
            all_predictions.append(predictions)
            all_targets.append(rotation_angle)

            # r2_metric.update(predictions, rotation_angle)

            # Log images and predictions
            if i % print_freq == 0:
                if wandb_log:
                    images = []
                    for j in range(input1.size(0)):
                        img = wandb.Image(
                            make_grid([input1[j].cpu(), input2[j].cpu()], nrow=2),
                            caption=f"Pred: {predictions[j].item()  :.2f}°, Actual: {rotation_angle[j].item() :.2f}°"
                        )
                        images.append(img)
                    wandb.log({"Test Images": images})

    test_loss = running_loss / len(test_loader)

    all_predictions = torch.cat(all_predictions).cpu()
    all_targets = torch.cat(all_targets).cpu()

    val_mae = torch.mean(torch.abs(all_predictions.float() - all_targets.float())).item()
    # test_r2 = r2_metric.compute().item()
    print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')


def run_pipeline(args):
    if args.wandb_log:
        proj_name = 'siamese-network-rotation-prediction-train' if args.do_train else 'siamese-network-rotation-prediction-test'
        wandb.init(project=proj_name)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet3D().to(device)

    if args.load_pretrained_weights:
        model_save_path = os.path.join(args.load_model_path, 'resnet3D_checkpoint_epoch_' + str(
            args.load_epoch_weights) + '.pth')  # Update with the correct path
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"Loaded pretrained weights from {model_save_path}")
        else:
            raise FileNotFoundError(f"No pretrained weights found at {args.model_save_path}")

    if args.do_train:
        train(model, epochs=args.num_epochs, save_frequency=args.save_freq, print_freq=args.print_freq,
              batch_size=args.batch_size, root_dir=args.root_dir, num_workers=args.num_workers,
              wandb_log=args.wandb_log)
    else:
        test(model, print_freq=args.print_freq, batch_size=args.batch_size, root_dir=args.root_dir,
             wandb_log=args.wandb_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--root_dir", type=str, default="./", help="Root directory for the project")
    parser.add_argument("--save_freq", type=int, default=1, help="Frequency of saving the model weights")
    parser.add_argument("--print_freq", type=int, default=100, help="Frequency of printing the training progress")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model training")
    parser.add_argument("--load_pretrained_weights", type=int, default=0, help="If True load a pretrained model")
    parser.add_argument("--load_model_path", type=str, default="./weights/", help="Path where saved weights are stored")
    parser.add_argument("--load_epoch_weights", type=int, default=1, help="Epoch weights to load for pre-trained model")
    parser.add_argument("--do_train", action="store_true", help="Enable training mode")
    parser.add_argument("--wandb_log", type=int, default=1, help="Enable or disable Weights & Biases logging")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")

    args = parser.parse_args()

    run_pipeline(args)
