#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for inference using a ResNet3D model for the task of angle prediction.

Contact Information:
    Name: Danai Triantafyllidou
    Email: danaitri22@gmail.com
    Date: 25-February-2025
"""

from torchvision import transforms, models
import torch
import torch.nn as nn
from PIL import Image
import argparse
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


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


def inference(filename_1: str, filename_2: str, model_save_path: str) -> None:
    """
    Perform inference on a pair of images using a pre-trained ResNet3D model.

    Args:
        filename_1 (str): Path to the first image file.
        filename_2 (str): Path to the second image file.
        model_save_path (str): Path to the saved model weights.

    Raises:
        FileNotFoundError: If the model weights file does not exist.
    """

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet3D().to(device)
    model.eval()

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded pretrained weights from {model_save_path}")
    else:
        raise FileNotFoundError(f"No pretrained weights found at {args.model_save_path}")

    image_1 = transform(Image.open(filename_1).convert('RGB')).unsqueeze(0).to(device)
    image_2 = transform(Image.open(filename_2).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_1, image_2)

    prediction = torch.argmax(output)
    print(f'Filename 1: {filename_1}, Filename 2: {filename_2}, Prediction: {prediction:.4f}')


def run_pipeline(args):
    inference(args.filename_1, args.filename_2, args.model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str, default="./weights/resnet3D_checkpoint_epoch_28.pth")
    parser.add_argument("--filename_1", type=str,
                        default="./data/coco2017/test2017_image_pairs/000000002884_original.jpg")
    parser.add_argument("--filename_2", type=str,
                        default="./data/coco2017/test2017_image_pairs/000000002884_rotated_188.59.jpg")

    args = parser.parse_args()

    run_pipeline(args)
