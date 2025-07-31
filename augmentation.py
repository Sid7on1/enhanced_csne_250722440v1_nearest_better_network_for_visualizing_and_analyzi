import logging
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Augmentation:
    """
    Class for data augmentation techniques.

    ...

    Attributes
    ----------
    transform_config : dict
        Configuration for data transformations.

    Methods
    -------
    apply_transforms(self, image, transform_type):
        Apply specified transforms to the input image.
    augment_dataset(self, dataset, transform_type):
        Augment the input dataset using the specified transform_type.
    """
    def __init__(self, transform_config):
        """
        Initialize the Augmentation class with the transform configuration.

        Parameters
        ----------
        transform_config : dict
            Configuration for data transformations, including the types of transformations and their parameters.
        """
        self.transform_config = transform_config

    def apply_transforms(self, image, transform_type):
        """
        Apply specified transforms to the input image.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor.
        transform_type : str
            Type of transformation to apply (e.g., 'random_crop', 'horizontal_flip').

        Returns
        -------
        torch.Tensor
            Transformed image.
        """
        try:
            if transform_type == 'random_crop':
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(32, 32))
                image = transforms.functional.crop(image, i, j, h, w)
            elif transform_type == 'horizontal_flip':
                if random.random() < 0.5:
                    image = transforms.functional.hflip(image)
            elif transform_type == 'color_jitter':
                image = transforms.functional.adjust_brightness(image, brightness_factor=random.uniform(0.5, 1.5))
                image = transforms.functional.adjust_contrast(image, contrast_factor=random.uniform(0.5, 1.5))
                image = transforms.functional.adjust_saturation(image, saturation_factor=random.uniform(0.5, 1.5))
                image = transforms.functional.adjust_hue(image, hue_factor=random.uniform(-0.5, 0.5))
            else:
                raise ValueError(f"Invalid transform_type: {transform_type}")

            return image

        except Exception as e:
            logging.error(f"Error applying transforms: {e}")
            raise

    def augment_dataset(self, dataset, transform_type):
        """
        Augment the input dataset using the specified transform_type.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Input dataset to be augmented.
        transform_type : str
            Type of transformation to apply (e.g., 'random_crop', 'horizontal_flip').

        Returns
        -------
        torch.utils.data.Dataset
            Augmented dataset.
        """
        try:
            augmented_dataset = CIFAR10(root='./data', train=True, download=True, transform=None)
            augmented_dataset.transform = self.transform_config[transform_type]
            return augmented_dataset

        except Exception as e:
            logging.error(f"Error augmenting dataset: {e}")
            raise

class AugmentationTrainer:
    """
    Class for training with data augmentation.

    ...

    Attributes
    ----------
    model : torch.nn.Module
        Neural network model to be trained.
    device : torch.device
        Device to use for training (CPU or GPU).
    optimizer : torch.optim.Optimizer
        Optimizer for updating model weights.
    loss_fn : callable
        Loss function for training.
    augmentation : Augmentation
        Augmentation object for data transformation.
    transform_types : list
        List of transformation types to apply.

    Methods
    -------
    train(self, train_loader, epochs):
        Train the model using the augmented dataset.
    """
    def __init__(self, model, device, optimizer, loss_fn, augmentation, transform_types):
        """
        Initialize the AugmentationTrainer class.

        Parameters
        ----------
        model : torch.nn.Module
            Neural network model to be trained.
        device : torch.device
            Device to use for training (CPU or GPU).
        optimizer : torch.optim.Optimizer
            Optimizer for updating model weights.
        loss_fn : callable
            Loss function for training.
        augmentation : Augmentation
            Augmentation object for data transformation.
        transform_types : list
            List of transformation types to apply.
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.augmentation = augmentation
        self.transform_types = transform_types

    def train(self, train_loader, epochs):
        """
        Train the model using the augmented dataset.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Data loader for the training dataset.
        epochs : int
            Number of epochs to train for.

        Returns
        -------
        None
        """
        try:
            self.model.to(self.device)

            for epoch in range(epochs):
                epoch_loss = 0
                epoch_time = time.time()

                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                    if batch_idx % 100 == 0:
                        logging.info(f"Epoch [{epoch+1}/{epochs}] - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

                epoch_time = time.time() - epoch_time
                logging.info(f"Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f}s - Loss: {epoch_loss / len(train_loader):.4f}")

                # Apply data augmentation after each epoch
                for transform_type in self.transform_types:
                    augmented_dataset = self.augmentation.augment_dataset(CIFAR10(root='./data', train=True, download=True, transform=None), transform_type)
                    train_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)

        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise

# Example usage
if __name__ == '__main__':
    transform_config = {
        'random_crop': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'horizontal_flip': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'color_jitter': transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    augmentation = Augmentation(transform_config)

    model = Net()  # Example model, replace with your model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    loss_fn = nn.CrossEntropyLoss()

    augmentation_trainer = AugmentationTrainer(model, device, optimizer, loss_fn, augmentation, ['random_crop', 'horizontal_flip', 'color_jitter'])

    train_loader = DataLoader(CIFAR10(root='./data', train=True, download=True, transform=transform_config['random_crop']), batch_size=64, shuffle=True)

    augmentation_trainer.train(train_loader, epochs=10)