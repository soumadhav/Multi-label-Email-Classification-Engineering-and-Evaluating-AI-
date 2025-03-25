"""
Base Model Module
Defines the abstract base class for all ML models
"""

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    This class defines the interface that all concrete model implementations must follow.
    It ensures that every model implements the same set of methods, providing a
    consistent interface regardless of the underlying model implementation.
    """
    
    def __init__(self) -> None:
        """Initialize the model"""
        ...


    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        
        This abstract method must be implemented by all concrete model classes.
        
        Args:
            data: Data object containing training data
        """
        ...

    @abstractmethod
    def predict(self) -> None:
        """
        Make predictions on test data.
        
        This abstract method must be implemented by all concrete model classes.
        
        Args:
            X_test: Test data features
        """
        ...

    @abstractmethod
    def print_results(self) -> None:
        """
        Print and evaluate model results.
        
        This abstract method must be implemented by all concrete model classes.
        
        Args:
            data: Data object containing test data
        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        """
        Transform data if needed by the model.
        
        This abstract method must be implemented by all concrete model classes.
        """
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
