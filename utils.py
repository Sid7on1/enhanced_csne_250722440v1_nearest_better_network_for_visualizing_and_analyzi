# utils.py
"""
Utility functions for the computer_vision project.
"""

import logging
import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from config import Config
from constants import Constants
from exceptions import (
    InvalidInputError,
    InvalidParameterError,
    InvalidStateError,
)
from metrics import Metrics
from utils_math import (
    calculate_velocity_threshold,
    calculate_flow_theory,
)
from utils_logging import (
    configure_logging,
    get_logger,
)
from utils_validation import (
    validate_input,
    validate_parameter,
    validate_state,
)

# Configure logging
configure_logging()

# Get the logger
logger = get_logger(__name__)

class Utils:
    """
    Utility functions for the computer_vision project.
    """

    def __init__(self, config: Config):
        """
        Initialize the Utils class.

        Args:
            config (Config): The project configuration.
        """
        self.config = config

    def calculate_nbn(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the Nearest-Better Network (NBN) for the given data.

        Args:
            data (np.ndarray): The input data.

        Returns:
            np.ndarray: The calculated NBN.
        """
        try:
            # Validate the input data
            validate_input(data)

            # Calculate the NBN using the velocity-threshold method
            velocity_threshold = calculate_velocity_threshold(data)
            nbn = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if data[i, j] > velocity_threshold:
                        nbn[i, j] = 1

            # Return the calculated NBN
            return nbn

        except InvalidInputError as e:
            # Log the error and re-raise it
            logger.error(f"Invalid input: {e}")
            raise

        except Exception as e:
            # Log the error and re-raise it
            logger.error(f"An error occurred: {e}")
            raise

    def calculate_flow_theory(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the Flow Theory for the given data.

        Args:
            data (np.ndarray): The input data.

        Returns:
            np.ndarray: The calculated Flow Theory.
        """
        try:
            # Validate the input data
            validate_input(data)

            # Calculate the Flow Theory using the Flow Theory method
            flow_theory = calculate_flow_theory(data)
            return flow_theory

        except InvalidInputError as e:
            # Log the error and re-raise it
            logger.error(f"Invalid input: {e}")
            raise

        except Exception as e:
            # Log the error and re-raise it
            logger.error(f"An error occurred: {e}")
            raise

    def get_metrics(self, data: np.ndarray) -> Metrics:
        """
        Get the metrics for the given data.

        Args:
            data (np.ndarray): The input data.

        Returns:
            Metrics: The calculated metrics.
        """
        try:
            # Validate the input data
            validate_input(data)

            # Calculate the metrics using the Metrics class
            metrics = Metrics(data)
            return metrics

        except InvalidInputError as e:
            # Log the error and re-raise it
            logger.error(f"Invalid input: {e}")
            raise

        except Exception as e:
            # Log the error and re-raise it
            logger.error(f"An error occurred: {e}")
            raise

def main():
    # Get the project configuration
    config = Config()

    # Create an instance of the Utils class
    utils = Utils(config)

    # Get some sample data
    data = np.random.rand(10, 10)

    # Calculate the NBN for the data
    nbn = utils.calculate_nbn(data)

    # Calculate the Flow Theory for the data
    flow_theory = utils.calculate_flow_theory(data)

    # Get the metrics for the data
    metrics = utils.get_metrics(data)

    # Print the results
    print("NBN:", nbn)
    print("Flow Theory:", flow_theory)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()