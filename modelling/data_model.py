"""
Data Model Module
Defines the Data class for encapsulating training and testing data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random

# Set random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

class Data:
    """
    Data class encapsulates all required input data in one object.
    This allows passing data elements consistently to all ML models.
    """
    
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        """
        Initialize Data object with feature matrix X and labels from dataframe
        
        Args:
            X (np.ndarray): Feature matrix (embeddings)
            df (pd.DataFrame): Dataframe containing labels
        """
        # For multi-label classification, we need to handle all types
        y2 = df.y2.to_numpy()
        y2Series = pd.Series(y2)
        
        # Filter out classes with too few samples
        goodY2Values = y2Series.value_counts()[y2Series.value_counts() >= 3].index
        
        if len(goodY2Values) < 1:
            print("None of the Type 2 classes have more than 3 records: Skipping...")
            self.X_train = None
            return
            
        # Get mask for records with valid y2 values
        maskGoodY2 = y2Series.isin(goodY2Values)
        
        # Filter data based on good y2 values
        X_good = X[maskGoodY2]
        y2_good = y2[maskGoodY2]
        
        # Get corresponding y3 and y4 values
        y3 = df.y3.to_numpy()
        y4 = df.y4.to_numpy()
        
        # Filter y3 and y4 with the same mask
        y3_good = y3[maskGoodY2]
        y4_good = y4[maskGoodY2]
        
        # Replace NaN values and empty strings with 'unknown' to avoid issues
        y3_good = np.array([
            'unknown' if pd.isna(val) or val == '' else val 
            for val in y3_good
        ], dtype=object)
        
        y4_good = np.array([
            'unknown' if pd.isna(val) or val == '' else val 
            for val in y4_good
        ], dtype=object)
        
        # Calculate new test size based on filtering
        newTestSize = X.shape[0] * 0.2 / X_good.shape[0]
        
        # Split data for training and testing with stratification
        self.X_train, self.X_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = train_test_split(
            X_good, y2_good, y3_good, y4_good, 
            test_size=newTestSize, 
            random_state=0, 
            stratify=y2_good
        )
        
        # Store all target variables
        self.y_train_dict = {
            'y2': y2_train,
            'y3': y3_train,
            'y4': y4_train
        }
        
        self.y_test_dict = {
            'y2': y2_test,
            'y3': y3_test,
            'y4': y4_test
        }
        
        # For backward compatibility
        self.y = y2_good
        self.y_train = y2_train
        self.y_test = y2_test
        self.classes = goodY2Values
        self.embeddings = X

    def get_type(self):
        """Get target variable for Type 2"""
        return self.y
        
    def get_X_train(self):
        """Get training features"""
        return self.X_train
        
    def get_X_test(self):
        """Get testing features"""
        return self.X_test
        
    def get_type_y_train(self):
        """Get training labels for Type 2"""
        return self.y_train
        
    def get_type_y_test(self):
        """Get testing labels for Type 2"""
        return self.y_test
        
    def get_embeddings(self):
        """Get all embeddings"""
        return self.embeddings
        
    def get_train_df(self):
        return self.train_df
        
    def get_type_test_df(self):
        return self.test_df
        
    def get_X_DL_test(self):
        return self.X_DL_test
        
    def get_X_DL_train(self):
        return self.X_DL_train

