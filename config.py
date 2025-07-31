import logging
import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# Define global constants
MODEL_CONFIG_PATH = "model_config.yaml"

# Type aliases
ModelType = Union[str, torch.nn.Module]
LossFunction = Union[str, torch.nn.Module]
MetricFunction = Union[str, torch.nn.Module]

# Type hints for function signatures
Device = Optional[Union[torch.device, str]]


class ModelConfig:
    """
    Model configuration class.

    Loads and manages configuration settings for the model.

    ...

    Attributes
    ----------
    config_path : str
        Path to the model configuration file.
    config : dict
        Dictionary containing the loaded configuration settings.
    device : torch.device
        Device to use for model training/inference (cpu or cuda).
    model : ModelType
        The model architecture to use. Can be either a string (name of a built-in model) or a custom model class.
    loss_fn : LossFunction
        The loss function to use during training. Can be either a string (name of a built-in loss function) or a custom loss function class.
    metrics : List[MetricFunction]
        A list of metric functions to use for evaluating the model. Each metric can be either a string (name of a built-in metric) or a custom metric class.
    optimizer : str
        The optimizer to use for model training.
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Batch size for training and inference.
    num_epochs : int
        Number of epochs to train the model.
    seed : int
        Random seed value for reproducibility.
    """

    def __init__(
        self,
        config_path: str = MODEL_CONFIG_PATH,
        device: Device = None,
    ) -> None:
        """
        Initializes the ModelConfig object by loading the configuration file and setting default values.

        Parameters
        ----------
        config_path : str, optional
            Path to the model configuration file. Defaults to MODEL_CONFIG_PATH.
        device : Union[torch.device, str], optional
            Device to use for model training/inference. Defaults to 'cuda' if a GPU is available, otherwise 'cpu'.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        ValueError
            If the configuration file is not in valid YAML format.
        """
        self.config_path = config_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load and validate the configuration file
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise ValueError(
                    "Invalid YAML format in configuration file."
                ) from e

        # Set default values for any missing settings
        default_config = {
            "model": "resnet50",
            "loss_fn": "cross_entropy",
            "metrics": ["accuracy"],
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 10,
            "seed": 42,
        }
        self.config = {**default_config, **self.config}

        # Validate the loaded configuration
        self._validate_config()

        # Set the random seed
        self._set_seed()

    def _validate_config(self) -> None:
        """
        Validates the loaded configuration settings and raises errors for any invalid values.
        """
        valid_models = ["resnet50", "vgg16", "custom_model"]
        if self.config["model"] not in valid_models:
            raise ValueError(
                f"Invalid model specified: {self.config['model']}. Choose from {valid_models}"
            )

        valid_loss_fns = ["mse", "cross_entropy", "custom_loss"]
        if self.config["loss_fn"] not in valid_loss_fns:
            raise ValueError(
                f"Invalid loss function specified: {self.config['loss_fn']}. Choose from {valid_loss_fns}"
            )

        valid_optimizers = ["adam", "sgd", "rmsprop"]
        if self.config["optimizer"] not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer specified: {self.config['optimizer']}. Choose from {valid_optimizers}"
            )

        if not isinstance(self.config["learning_rate"], (int, float)):
            raise ValueError("Learning rate must be a number.")
        if not isinstance(self.config["batch_size"], int):
            raise ValueError("Batch size must be an integer.")
        if not isinstance(self.config["num_epochs"], int):
            raise ValueError("Number of epochs must be an integer.")
        if not isinstance(self.config["seed"], int):
            raise ValueError("Seed must be an integer.")

    def _set_seed(self) -> None:
        """
        Sets the random seed for reproducibility.
        """
        torch.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])

    @property
    def model(self) -> ModelType:
        """
        Returns the model architecture to use.

        Returns
        ----------
        ModelType
            The model architecture, either as a string (name of a built-in model) or a custom model class.
        """
        return self.config["model"]

    @property
    def loss_fn(self) -> LossFunction:
        """
        Returns the loss function to use during training.

        Returns
        ----------
        LossFunction
            The loss function, either as a string (name of a built-in loss function) or a custom loss function class.
        """
        return self.config["loss_fn"]

    @property
    def metrics(self) -> List[MetricFunction]:
        """
        Returns a list of metric functions to use for evaluating the model.

        Returns
        ----------
        List[MetricFunction]
            A list of metric functions, each either as a string (name of a built-in metric) or a custom metric class.
        """
        return self.config["metrics"]

    @property
    def optimizer(self) -> str:
        """
        Returns the optimizer to use for model training.

        Returns
        ----------
        str
            The name of the optimizer.
        """
        return self.config["optimizer"]

    @property
    def learning_rate(self) -> float:
        """
        Returns the learning rate for the optimizer.

        Returns
        ----------
        float
            The learning rate.
        """
        return self.config["learning_rate"]

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size for training and inference.

        Returns
        ----------
        int
            The batch size.
        """
        return self.config["batch_size"]

    @property
    def num_epochs(self) -> int:
        """
        Returns the number of epochs to train the model.

        Returns
        ----------
        int
            The number of epochs.
        """
        return self.config["num_epochs"]

    def get_dataloader(
        self,
        dataset: Union[torch.utils.data.Dataset, str],
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ) -> DataLoader:
        """
        Creates a PyTorch DataLoader for the given dataset.

        Parameters
        ----------
        dataset : Union[torch.utils.data.Dataset, str]
            The dataset to create the DataLoader for. Can be either a PyTorch Dataset object or the name of a built-in dataset.
        batch_size : int, optional
            Batch size for the DataLoader. Defaults to the value specified in the config if not provided.
        shuffle : bool, optional
            Whether to shuffle the data before creating batches. Defaults to False.
        drop_last : bool, optional
            Whether to drop the last incomplete batch. Defaults to False.
        **kwargs : Any
            Additional keyword arguments to pass to the DataLoader constructor.

        Returns
        ----------
        DataLoader
            The created DataLoader object.

        Raises
        ------
        ValueError
            If the provided dataset is not recognized.
        """
        batch_size = batch_size or self.batch_size

        if isinstance(dataset, str):
            if dataset == "cifar10":
                from torchvision.datasets import CIFAR10

                dataset = CIFAR10
            elif dataset == "mnist":
                from torchvision.datasets import MNIST

                dataset = MNIST
            else:
                raise ValueError(f"Unrecognized dataset: {dataset}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )

        return dataloader

    def get_model(self, num_classes: int) -> nn.Module:
        """
        Returns the model architecture based on the configuration.

        Parameters
        ----------
        num_classes : int
            Number of output classes for the model.

        Returns
        ----------
        nn.Module
            The model architecture, either a built-in model or a custom model class.

        Raises
        ------
        ValueError
            If the specified model is not recognized.
        """
        model_name = self.config["model"]

        if model_name == "resnet50":
            from torchvision.models import resnet50

            model = resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)

        elif model_name == "vgg16":
            from torchvision.models import vgg16

            model = vgg16(pretrained=True)
            num_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_features, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

        elif model_name == "custom_model":
            # Example custom model
            model = CustomModel(num_classes)

        else:
            raise ValueError(f"Unrecognized model: {model_name}")

        model.to(self.device)

        return model

    def get_loss_fn(self) -> LossFunction:
        """
        Returns the loss function based on the configuration.

        Returns
        ----------
        LossFunction
            The loss function, either a built-in loss function or a custom loss function class.

        Raises
        ------
        ValueError
            If the specified loss function is not recognized.
        """
        loss_fn_name = self.config["loss_fn"]

        if loss_fn_name == "mse":
            loss_fn = nn.MSELoss()
        elif loss_fn_name == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
        elif loss_fn_name == "custom_loss":
            from custom_loss import CustomLoss

            loss_fn = CustomLoss()
        else:
            raise ValueError(f"Unrecognized loss function: {loss_fn_name}")

        return loss_fn

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Returns the optimizer based on the configuration.

        Parameters
        ----------
        model : nn.Module
            The model to optimize.

        Returns
        ----------
        torch.optim.Optimizer
            The optimizer to use for training the model.

        Raises
        ------
        ValueError
            If the specified optimizer is not recognized.
        """
        optimizer_name = self.config["optimizer"]

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unrecognized optimizer: {optimizer_name}")

        return optimizer

    def get_learning_rate_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        steps_per_epoch: int,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Returns a learning rate scheduler based on the configuration.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to use.
        num_epochs : int
            Total number of epochs for training.
        steps_per_epoch : int
            Number of steps (batches) per epoch.

        Returns
        ----------
        Optional[torch.optim.lr_scheduler._LRScheduler]
            The learning rate scheduler, or None if no scheduler is specified in the config.
        """
        scheduler = None
        scheduler_config = self.config.get("learning_rate_scheduler", {})
        scheduler_type = scheduler_config.get("type", None)

        if scheduler_type == "step":
            step_size = scheduler_config.get("step_size", 10)
            gamma = scheduler_config.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "cosine":
            t_max = scheduler_config.get("t_max", num_epochs * steps_per_epoch)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max
            )

        return scheduler


class CustomModel(nn.Module):
    """
    Example custom model class.

    This is just a simple example model with two linear layers.
    Replace this with your own custom model architecture.
    """

    def __init__(self, num_classes: int):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


# Example usage
if __name__ == "__main__":
    config = ModelConfig()
    print(config.model)
    print(config.loss_fn)
    print(config.metrics)
    print(config.optimizer)
    print(config.learning_rate)
    print(config.batch_size)
    print(config.num_epochs)