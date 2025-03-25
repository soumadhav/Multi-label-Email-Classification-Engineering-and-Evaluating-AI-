import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import LabelEncoder
from Config import *
import random

seed = 0
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class ChainedModel(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: dict) -> None:
        """
        Initialize ChainedModel with embeddings and multiple target variables
        
        Args:
            model_name: Name of the model
            embeddings: Feature vectors
            y: Dictionary of target variables (y2, y3, y4)
        """
        super(ChainedModel, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        
        # Initialize base classifier
        self.base_classifier = RandomForestClassifier(
            n_estimators=1000, 
            random_state=seed, 
            class_weight='balanced_subsample'
        )
        
        # Create separate models for each level instead of using chains
        self.model_y2 = RandomForestClassifier(
            n_estimators=1000, 
            random_state=seed, 
            class_weight='balanced_subsample'
        )
        
        self.model_y23 = RandomForestClassifier(
            n_estimators=1000, 
            random_state=seed, 
            class_weight='balanced_subsample'
        )
        
        self.model_y234 = RandomForestClassifier(
            n_estimators=1000, 
            random_state=seed, 
            class_weight='balanced_subsample'
        )
        
        # Label encoders to convert categorical labels to numeric
        self.y2_encoder = LabelEncoder()
        self.y3_encoder = LabelEncoder()
        self.y4_encoder = LabelEncoder()
        
        self.predictions_y2 = None
        self.predictions_y3 = None
        self.predictions_y4 = None
        self.data_transform()

    def train(self, data) -> None:
        """
        Train the models for each level of prediction
        
        Args:
            data: Data object containing training data with multiple targets
        """
        # Encode training labels
        y2_encoded = self.y2_encoder.fit_transform(data.y_train_dict['y2'])
        
        # Handle empty strings in y3 and y4 - convert to 'unknown'
        y3_train = np.array([val if val != '' else 'unknown' for val in data.y_train_dict['y3']])
        y4_train = np.array([val if val != '' else 'unknown' for val in data.y_train_dict['y4']])
        
        # Fit label encoders
        y3_encoded = self.y3_encoder.fit_transform(y3_train)
        y4_encoded = self.y4_encoder.fit_transform(y4_train)
        
        # Print class distributions for monitoring
        print(f"\nClass distribution for Type 2: {np.bincount(y2_encoded)}")
        print(f"Class distribution for Type 3: {np.bincount(y3_encoded)}")
        print(f"Class distribution for Type 4: {np.bincount(y4_encoded)}")
        
        # Train the Type 2 model
        self.model_y2.fit(data.X_train, y2_encoded)
        
        # Create features for Type 2 + Type 3 model (include Type 2 predictions)
        X_train_y2_pred = self.model_y2.predict(data.X_train).reshape(-1, 1)
        X_train_y23 = np.hstack((data.X_train, X_train_y2_pred))
        self.model_y23.fit(X_train_y23, y3_encoded)
        
        # Create features for Type 2 + Type 3 + Type 4 model (include Type 2 and Type 3 predictions)
        X_train_y3_pred = self.model_y23.predict(X_train_y23).reshape(-1, 1)
        X_train_y234 = np.hstack((X_train_y23, X_train_y3_pred))
        self.model_y234.fit(X_train_y234, y4_encoded)

    def predict(self, X_test: np.ndarray):
        """
        Make predictions using the chain of models
        
        Args:
            X_test: Test data features
        """
        # Predict Type 2
        y2_pred_encoded = self.model_y2.predict(X_test)
        self.predictions_y2 = self.y2_encoder.inverse_transform(y2_pred_encoded)
        
        # Predict Type 3 using Type 2 predictions as additional features
        X_test_y2_pred = y2_pred_encoded.reshape(-1, 1)
        X_test_y23 = np.hstack((X_test, X_test_y2_pred))
        y3_pred_encoded = self.model_y23.predict(X_test_y23)
        self.predictions_y3 = self.y3_encoder.inverse_transform(y3_pred_encoded)
        
        # Predict Type 4 using Type 2 and Type 3 predictions as additional features
        X_test_y3_pred = y3_pred_encoded.reshape(-1, 1)
        X_test_y234 = np.hstack((X_test_y23, X_test_y3_pred))
        y4_pred_encoded = self.model_y234.predict(X_test_y234)
        self.predictions_y4 = self.y4_encoder.inverse_transform(y4_pred_encoded)

    def print_results(self, data):
        """
        Print classification results for all models
        
        Args:
            data: Data object containing test data with multiple targets
        """
        # Handle empty strings in test data
        y3_test = np.array([val if val != '' else 'unknown' for val in data.y_test_dict['y3']])
        y4_test = np.array([val if val != '' else 'unknown' for val in data.y_test_dict['y4']])
        
        # Print results for Type 2
        print("\n=== Chained Model: Classification Report for Type 2 ===")
        print(classification_report(data.y_test_dict['y2'], self.predictions_y2, zero_division=0))
        
        # Print results for Type 3 (using Type 2 predictions)
        print("\n=== Chained Model: Classification Report for Type 3 (with Type 2 predictions) ===")
        print(classification_report(y3_test, self.predictions_y3, zero_division=0))
        
        # Print results for Type 4 (using Type 2 and Type 3 predictions)
        print("\n=== Chained Model: Classification Report for Type 4 (with Type 2 and Type 3 predictions) ===")
        print(classification_report(y4_test, self.predictions_y4, zero_division=0))
        
        # Calculate individual accuracies
        type2_acc = np.mean(data.y_test_dict['y2'] == self.predictions_y2)
        type3_acc = np.mean(y3_test == self.predictions_y3)
        type4_acc = np.mean(y4_test == self.predictions_y4)
        
        print(f"\nType 2 accuracy: {type2_acc}")
        print(f"Type 3 accuracy: {type3_acc}")
        print(f"Type 4 accuracy: {type4_acc}")
        
        # Calculate overall accuracy (all levels correct)
        correct_y2 = (data.y_test_dict['y2'] == self.predictions_y2)
        correct_y3 = (y3_test == self.predictions_y3)
        correct_y4 = (y4_test == self.predictions_y4)
        
        overall_acc = np.mean(correct_y2 & correct_y3 & correct_y4)
        print(f"\nOverall accuracy (all levels correct): {overall_acc}")

    def data_transform(self) -> None:
        """Data transformation if needed"""
        pass 