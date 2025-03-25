import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from Config import *
import random
from collections import defaultdict

seed = 0
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class HierarchicalModel(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: dict) -> None:
        """
        Initialize HierarchicalModel with embeddings and multiple target variables
        
        Args:
            model_name: Name of the model
            embeddings: Feature vectors
            y: Dictionary of target variables (y2, y3, y4)
        """
        super(HierarchicalModel, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        
        # Models at each level
        self.level1_model = None  # Type 2 classifier
        self.level2_models = {}   # Type 3 classifiers (one per Type 2 class)
        self.level3_models = {}   # Type 4 classifiers (one per Type 2+Type 3 combination)
        
        # Predictions
        self.predictions_y2 = None
        self.predictions_y3 = None
        self.predictions_y4 = None
        
        # Track unique classes at each level
        self.y2_classes = None
        self.y3_classes = {}  # For each Type 2 class
        self.y4_classes = {}  # For each Type 2+Type 3 combination
        
        self.data_transform()

    def train(self, data) -> None:
        """
        Train hierarchical models:
        1. Train Type 2 classifier
        2. For each Type 2 class, train a Type 3 classifier
        3. For each Type 2+Type 3 combination, train a Type 4 classifier
        
        Args:
            data: Data object containing training data with multiple targets
        """
        # Get unique classes for Type 2
        self.y2_classes = np.unique(data.y_train_dict['y2'])
        
        # Train Level 1 model (Type 2)
        self.level1_model = RandomForestClassifier(
            n_estimators=1000, 
            random_state=seed, 
            class_weight='balanced_subsample'
        )
        self.level1_model.fit(data.X_train, data.y_train_dict['y2'])
        
        # For each Type 2 class, train a Type 3 classifier
        for y2_class in self.y2_classes:
            # Filter training data for this Type 2 class
            mask = data.y_train_dict['y2'] == y2_class
            if np.sum(mask) < 3:  # Skip if too few samples
                continue
                
            X_train_filtered = data.X_train[mask]
            y3_train_filtered = data.y_train_dict['y3'][mask]
            
            # Replace empty strings with 'unknown'
            y3_train_filtered = np.array([val if val != '' else 'unknown' for val in y3_train_filtered], dtype=object)
            
            # Get unique Type 3 classes for this Type 2 class
            self.y3_classes[y2_class] = np.unique(y3_train_filtered)
            
            # Train Type 3 classifier for this Type 2 class
            if len(self.y3_classes[y2_class]) > 1:  # Only if multiple classes exist
                model = RandomForestClassifier(
                    n_estimators=1000, 
                    random_state=seed, 
                    class_weight='balanced_subsample'
                )
                model.fit(X_train_filtered, y3_train_filtered)
                self.level2_models[y2_class] = model
        
        # For each Type 2+Type 3 combination, train a Type 4 classifier
        for y2_class in self.y2_classes:
            if y2_class not in self.y3_classes:
                continue
                
            for y3_class in self.y3_classes[y2_class]:
                # Filter training data for this Type 2+Type 3 combination
                mask = (data.y_train_dict['y2'] == y2_class) & (data.y_train_dict['y3'] == y3_class)
                if np.sum(mask) < 3:  # Skip if too few samples
                    continue
                    
                X_train_filtered = data.X_train[mask]
                y4_train_filtered = data.y_train_dict['y4'][mask]
                
                # Replace empty strings with 'unknown'
                y4_train_filtered = np.array([val if val != '' else 'unknown' for val in y4_train_filtered], dtype=object)
                
                # Get unique Type 4 classes for this Type 2+Type 3 combination
                key = (y2_class, y3_class)
                self.y4_classes[key] = np.unique(y4_train_filtered)
                
                # Train Type 4 classifier for this Type 2+Type 3 combination
                if len(self.y4_classes[key]) > 1:  # Only if multiple classes exist
                    model = RandomForestClassifier(
                        n_estimators=1000, 
                        random_state=seed, 
                        class_weight='balanced_subsample'
                    )
                    model.fit(X_train_filtered, y4_train_filtered)
                    self.level3_models[key] = model

    def predict(self, X_test: np.ndarray):
        """
        Make predictions using the hierarchical models:
        1. Predict Type 2
        2. For each instance, based on predicted Type 2, predict Type 3
        3. For each instance, based on predicted Type 2+Type 3, predict Type 4
        
        Args:
            X_test: Test data features
        """
        n_samples = X_test.shape[0]
        
        # Predict Type 2
        self.predictions_y2 = self.level1_model.predict(X_test)
        
        # Initialize Type 3 and Type 4 predictions (with default values)
        self.predictions_y3 = np.array([''] * n_samples, dtype=object)
        self.predictions_y4 = np.array([''] * n_samples, dtype=object)
        
        # For each instance, predict Type 3 based on predicted Type 2
        for i in range(n_samples):
            y2_pred = self.predictions_y2[i]
            
            # If we have a Type 3 model for this Type 2 class
            if y2_pred in self.level2_models:
                model = self.level2_models[y2_pred]
                self.predictions_y3[i] = model.predict(X_test[i:i+1])[0]
            elif y2_pred in self.y3_classes and len(self.y3_classes[y2_pred]) == 1:
                # If only one Type 3 class for this Type 2 class, use it
                self.predictions_y3[i] = self.y3_classes[y2_pred][0]
        
        # For each instance, predict Type 4 based on predicted Type 2+Type 3
        for i in range(n_samples):
            y2_pred = self.predictions_y2[i]
            y3_pred = self.predictions_y3[i]
            
            if y3_pred == '':  # Skip if Type 3 prediction is empty
                continue
                
            key = (y2_pred, y3_pred)
            
            # If we have a Type 4 model for this Type 2+Type 3 combination
            if key in self.level3_models:
                model = self.level3_models[key]
                self.predictions_y4[i] = model.predict(X_test[i:i+1])[0]
            elif key in self.y4_classes and len(self.y4_classes[key]) == 1:
                # If only one Type 4 class for this Type 2+Type 3 combination, use it
                self.predictions_y4[i] = self.y4_classes[key][0]

    def print_results(self, data):
        """
        Print classification results for hierarchical models
        
        Args:
            data: Data object containing test data with multiple targets
        """
        # Evaluate Type 2 predictions
        print("\n=== Hierarchical Model: Classification Report for Type 2 ===")
        print(classification_report(data.y_test_dict['y2'], self.predictions_y2, zero_division=0))
        
        # Evaluate Type 3 predictions where Type 2 is correct
        mask_correct_y2 = data.y_test_dict['y2'] == self.predictions_y2
        
        if np.sum(mask_correct_y2) > 0:
            print("\n=== Hierarchical Model: Classification Report for Type 3 (where Type 2 is correct) ===")
            type3_acc = classification_report(
                data.y_test_dict['y3'][mask_correct_y2],
                self.predictions_y3[mask_correct_y2],
                output_dict=True,
                zero_division=0
            )['weighted avg']['precision']
            print(f"Type 3 accuracy given correct Type 2: {type3_acc}")
        
        # Evaluate Type 4 predictions where Type 2 and Type 3 are correct
        mask_correct_y23 = mask_correct_y2 & (data.y_test_dict['y3'] == self.predictions_y3)
        
        if np.sum(mask_correct_y23) > 0:
            print("\n=== Hierarchical Model: Classification Report for Type 4 (where Type 2 and Type 3 are correct) ===")
            type4_acc = classification_report(
                data.y_test_dict['y4'][mask_correct_y23],
                self.predictions_y4[mask_correct_y23],
                output_dict=True,
                zero_division=0
            )['weighted avg']['precision']
            print(f"Type 4 accuracy given correct Type 2 and Type 3: {type4_acc}")
        
        # Overall accuracy (all 3 levels correct)
        mask_all_correct = mask_correct_y23 & (data.y_test_dict['y4'] == self.predictions_y4)
        overall_acc = np.sum(mask_all_correct) / len(self.predictions_y2)
        print(f"\nOverall accuracy (all levels correct): {overall_acc}")

    def data_transform(self) -> None:
        """Data transformation if needed"""
        pass 