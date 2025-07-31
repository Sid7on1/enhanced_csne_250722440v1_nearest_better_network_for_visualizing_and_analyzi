# -*- coding: utf-8 -*-

"""
Image preprocessing utilities
"""

import logging
import os
import sys
import numpy as np
import cv2
from typing import Tuple, List, Dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config
from utils import load_config, setup_logging

# Set up logging
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Image preprocessing utilities
    """

    def __init__(self, config: Config):
        """
        Initialize the image preprocessor

        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.transform = self._create_transform()

    def _create_transform(self) -> transforms.Compose:
        """
        Create a transform pipeline for image preprocessing

        Returns:
            transforms.Compose: Transform pipeline
        """
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])
        return transform

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess an image

        Args:
            image_path (str): Path to the image

        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed image and its label
        """
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transform(image)
            return image.numpy(), np.array([0])  # dummy label
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None, None

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image

        Args:
            image_path (str): Path to the image

        Returns:
            np.ndarray: Loaded image
        """
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save an image

        Args:
            image (np.ndarray): Image to save
            output_path (str): Output path
        """
        try:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image)
        except Exception as e:
            logger.error(f"Error saving image: {e}")

class ImageDataset(Dataset):
    """
    Custom dataset class for images
    """

    def __init__(self, image_paths: List[str], config: Config):
        """
        Initialize the dataset

        Args:
            image_paths (List[str]): List of image paths
            config (Config): Configuration object
        """
        self.image_paths = image_paths
        self.config = config
        self.transform = self._create_transform()

    def _create_transform(self) -> transforms.Compose:
        """
        Create a transform pipeline for image preprocessing

        Returns:
            transforms.Compose: Transform pipeline
        """
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])
        return transform

    def __len__(self) -> int:
        """
        Get the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get an item from the dataset

        Args:
            index (int): Index of the item

        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed image and its label
        """
        image_path = self.image_paths[index]
        image = self.load_image(image_path)
        image = self.transform(Image.fromarray(image))
        return image.numpy(), np.array([0])  # dummy label

def main():
    """
    Main function
    """
    config = load_config()
    setup_logging(config)
    preprocessor = ImagePreprocessor(config)
    image_path = "path/to/image.jpg"
    image, label = preprocessor.preprocess_image(image_path)
    print(image.shape)
    print(label)

if __name__ == "__main__":
    main()