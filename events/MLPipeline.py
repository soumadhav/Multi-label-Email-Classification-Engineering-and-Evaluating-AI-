"""
ML Pipeline with Event-Driven Architecture
Orchestrates the machine learning workflow through events
Supports asynchronous event processing for improved performance
"""

from Config import Config
from events.EventBus import EventBus
from preprocess import get_input_data, de_duplication, noise_remover
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from model.chained_model import ChainedModel
from model.hierarchical_model import HierarchicalModel
from model.randomforest import RandomForest

import random
import numpy as np
import pandas as pd
import traceback
import asyncio
from typing import Dict, Any, Optional

# Set random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

class MLPipeline:
    """
    Event-driven machine learning pipeline that coordinates the different
    processing stages through event publishing and subscriptions.
    
    Supports both synchronous and asynchronous event processing for improved performance.
    """
    
    def __init__(self, useAsync: bool = False):
        """
        Initialize the ML pipeline with an event bus and set up event handlers
        
        Args:
            useAsync (bool): Whether to use asynchronous event processing
        """
        self.eventBus = EventBus()
        self.trainedComponents = {}
        self.useAsync = useAsync
        
        # Set up event handlers/subscribers based on mode
        if useAsync:
            # Asynchronous subscribers
            self.eventBus.subscribeAsync(Config.EVENT_DATA_LOADED, self.preprocessDataAsync)
            self.eventBus.subscribeAsync(Config.EVENT_DATA_PREPROCESSED, self.createEmbeddingsAsync)
            self.eventBus.subscribeAsync(Config.EVENT_EMBEDDINGS_CREATED, self.transformDataAsync)
            self.eventBus.subscribeAsync(Config.EVENT_DATA_TRANSFORMED, self.trainModelAsync)
            self.eventBus.subscribeAsync(Config.EVENT_MODEL_TRAINED, self.makePredictionsAsync)
            self.eventBus.subscribeAsync(Config.EVENT_PREDICTIONS_MADE, self.evaluateModelAsync)
        else:
            # Synchronous subscribers
            self.eventBus.subscribe(Config.EVENT_DATA_LOADED, self.preprocessData)
            self.eventBus.subscribe(Config.EVENT_DATA_PREPROCESSED, self.createEmbeddings)
            self.eventBus.subscribe(Config.EVENT_EMBEDDINGS_CREATED, self.transformData)
            self.eventBus.subscribe(Config.EVENT_DATA_TRANSFORMED, self.trainModel)
            self.eventBus.subscribe(Config.EVENT_MODEL_TRAINED, self.makePredictions)
            self.eventBus.subscribe(Config.EVENT_PREDICTIONS_MADE, self.evaluateModel)
    
    # Synchronous methods
    def loadData(self) -> Dict[str, Any]:
        """
        Load the input data and publish data_loaded event
        
        Returns:
            Dict containing loaded data
        """
        print("Loading data...")
        try:
            # Load the input data
            df = get_input_data()
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Publish event with loaded data
            return self.eventBus.publish(Config.EVENT_DATA_LOADED, {"df": df})
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None
    
    # Asynchronous methods
    async def loadDataAsync(self) -> Dict[str, Any]:
        """
        Asynchronously load the input data and publish data_loaded event
        
        Returns:
            Dict containing loaded data
        """
        print("Loading data asynchronously...")
        try:
            # Run data loading in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, get_input_data)
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Publish event with loaded data
            return await self.eventBus.publishAsync(Config.EVENT_DATA_LOADED, {"df": df})
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None
    
    def preprocessData(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data and publish data_preprocessed event
        
        Args:
            data (dict): Dictionary containing the dataframe
            
        Returns:
            dict: Updated data dictionary with preprocessed dataframe
        """
        df = data["df"]
        if df is None:
            return None
        
        print("Preprocessing data...")
        try:
            # De-duplicate input data
            df = de_duplication(df)
            
            # Remove noise in input data
            df = noise_remover(df)
            
            # Ensure string columns are properly typed
            df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
            df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
            
            print("Data preprocessing completed.")
            
            # Publish event with preprocessed data
            return self.eventBus.publish(Config.EVENT_DATA_PREPROCESSED, {"df": df})
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            traceback.print_exc()
            return None
    
    async def preprocessDataAsync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously preprocess the data and publish data_preprocessed event
        
        Args:
            data (dict): Dictionary containing the dataframe
            
        Returns:
            dict: Updated data dictionary with preprocessed dataframe
        """
        df = data["df"]
        if df is None:
            return None
        
        print("Preprocessing data asynchronously...")
        try:
            loop = asyncio.get_event_loop()
            
            # Run preprocessing steps in thread pool
            df = await loop.run_in_executor(None, lambda: de_duplication(df))
            df = await loop.run_in_executor(None, lambda: noise_remover(df))
            
            # Ensure string columns are properly typed
            df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
            df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
            
            print("Data preprocessing completed.")
            
            # Publish event with preprocessed data
            return await self.eventBus.publishAsync(Config.EVENT_DATA_PREPROCESSED, {"df": df})
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            traceback.print_exc()
            return None
    
    def createEmbeddings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create embeddings from preprocessed data and publish embeddings_created event
        
        Args:
            data (dict): Dictionary containing the preprocessed dataframe
            
        Returns:
            dict: Updated data dictionary with embeddings and dataframe
        """
        df = data["df"]
        if df is None:
            return None
        
        print("Creating embeddings...")
        try:
            # Get TF-IDF embeddings
            X = get_tfidf_embd(df)
            print(f"Embeddings created. Shape: {X.shape}")
            
            # Publish event with embeddings
            return self.eventBus.publish(Config.EVENT_EMBEDDINGS_CREATED, {"X": X, "df": df})
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            traceback.print_exc()
            return None
    
    async def createEmbeddingsAsync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously create embeddings from preprocessed data
        
        Args:
            data (dict): Dictionary containing the preprocessed dataframe
            
        Returns:
            dict: Updated data dictionary with embeddings and dataframe
        """
        df = data["df"]
        if df is None:
            return None
        
        print("Creating embeddings asynchronously...")
        try:
            # Run embedding creation in thread pool
            loop = asyncio.get_event_loop()
            X = await loop.run_in_executor(None, lambda: get_tfidf_embd(df))
            print(f"Embeddings created. Shape: {X.shape}")
            
            # Publish event with embeddings
            return await self.eventBus.publishAsync(Config.EVENT_EMBEDDINGS_CREATED, {"X": X, "df": df})
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            traceback.print_exc()
            return None
    
    def transformData(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data into model-ready format and publish data_transformed event
        
        Args:
            data (dict): Dictionary containing embeddings and dataframe
            
        Returns:
            dict: Updated data dictionary with Data object
        """
        X = data["X"]
        df = data["df"]
        
        print("Creating data object...")
        try:
            # Group data by type 1 (y1)
            print(f"Grouped by: {Config.GROUPED}")
            print(f"Unique values in {Config.GROUPED}: {df[Config.GROUPED].unique()}")
            
            groupedDf = df.groupby(Config.GROUPED)
            print(f"Number of groups: {len(groupedDf)}")
            
            # Store groups for processing
            data["groups"] = []
            
            for name, groupDf in groupedDf:
                print(f"\nProcessing group: {name}")
                print(f"Group size: {groupDf.shape}")
                
                # Filter embeddings for this group
                groupX = X[df[Config.GROUPED] == name]
                
                # Create Data object
                modelData = Data(groupX, groupDf)
                
                if modelData.X_train is not None:
                    data["groups"].append({
                        "name": name,
                        "data": modelData,
                        "df": groupDf
                    })
                else:
                    print(f"Skipping group {name} due to insufficient data")
            
            # Publish event with transformed data
            return self.eventBus.publish(Config.EVENT_DATA_TRANSFORMED, data)
        except Exception as e:
            print(f"Error transforming data: {e}")
            traceback.print_exc()
            return None
    
    async def transformDataAsync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously transform data into model-ready format
        
        Args:
            data (dict): Dictionary containing embeddings and dataframe
            
        Returns:
            dict: Updated data dictionary with Data object
        """
        X = data["X"]
        df = data["df"]
        
        print("Creating data object asynchronously...")
        try:
            # Group data by type 1 (y1)
            print(f"Grouped by: {Config.GROUPED}")
            print(f"Unique values in {Config.GROUPED}: {df[Config.GROUPED].unique()}")
            
            loop = asyncio.get_event_loop()
            groupedDf = df.groupby(Config.GROUPED)
            print(f"Number of groups: {len(groupedDf)}")
            
            # Store groups for processing
            data["groups"] = []
            
            for name, groupDf in groupedDf:
                print(f"\nProcessing group: {name}")
                print(f"Group size: {groupDf.shape}")
                
                # Filter embeddings for this group
                groupX = X[df[Config.GROUPED] == name]
                
                # Create Data object (run in thread pool if it's computationally expensive)
                modelData = await loop.run_in_executor(None, lambda: Data(groupX, groupDf))
                
                if modelData.X_train is not None:
                    data["groups"].append({
                        "name": name,
                        "data": modelData,
                        "df": groupDf
                    })
                else:
                    print(f"Skipping group {name} due to insufficient data")
            
            # Publish event with transformed data
            return await self.eventBus.publishAsync(Config.EVENT_DATA_TRANSFORMED, data)
        except Exception as e:
            print(f"Error transforming data: {e}")
            traceback.print_exc()
            return None
    
    def trainModel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train models for each group and publish model_trained event
        
        Args:
            data (dict): Dictionary containing groups with Data objects
            
        Returns:
            dict: Updated data dictionary with trained models
        """
        groups = data["groups"]
        
        print("Training models...")
        try:
            for group in groups:
                groupName = group["name"]
                groupData = group["data"]
                groupDf = group["df"]
                
                print(f"\nTraining models for group: {groupName}")
                
                # Create model instance based on classification approach
                if Config.CLASSIFICATION_APPROACH == Config.CHAINED_OUTPUT:
                    print("=== Running Chained Multi-Output Classification ===")
                    model = ChainedModel("ChainedModel", groupData.get_embeddings(), groupData.y_train_dict)
                    model.train(groupData)
                    group["model"] = model
                    
                elif Config.CLASSIFICATION_APPROACH == Config.HIERARCHICAL:
                    print("=== Running Hierarchical Modeling ===")
                    model = HierarchicalModel("HierarchicalModel", groupData.get_embeddings(), groupData.y_train_dict)
                    model.train(groupData)
                    group["model"] = model
                    
                else:
                    # Original single-label classification (for backward compatibility)
                    print("=== Running Single-Label Classification (RandomForest) ===")
                    model = RandomForest("RandomForest", groupData.get_embeddings(), groupData.get_type())
                    model.train(groupData)
                    group["model"] = model
            
            # Publish event with trained models
            return self.eventBus.publish(Config.EVENT_MODEL_TRAINED, data)
        except Exception as e:
            print(f"Error training models: {e}")
            traceback.print_exc()
            return None
    
    async def trainModelAsync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously train models for each group
        
        Uses parallel processing to train models for different groups simultaneously.
        
        Args:
            data (dict): Dictionary containing groups with Data objects
            
        Returns:
            dict: Updated data dictionary with trained models
        """
        groups = data["groups"]
        
        print("Training models asynchronously...")
        try:
            loop = asyncio.get_event_loop()
            
            # Prepare training tasks for each group
            async def train_group(group):
                groupName = group["name"]
                groupData = group["data"]
                
                print(f"\nTraining models for group: {groupName}")
                
                # Create and train model based on classification approach
                if Config.CLASSIFICATION_APPROACH == Config.CHAINED_OUTPUT:
                    print(f"=== Running Chained Multi-Output Classification for {groupName} ===")
                    model = ChainedModel("ChainedModel", groupData.get_embeddings(), groupData.y_train_dict)
                    # Run training in thread pool
                    await loop.run_in_executor(None, lambda: model.train(groupData))
                    group["model"] = model
                    
                elif Config.CLASSIFICATION_APPROACH == Config.HIERARCHICAL:
                    print(f"=== Running Hierarchical Modeling for {groupName} ===")
                    model = HierarchicalModel("HierarchicalModel", groupData.get_embeddings(), groupData.y_train_dict)
                    # Run training in thread pool
                    await loop.run_in_executor(None, lambda: model.train(groupData))
                    group["model"] = model
                    
                else:
                    # Original single-label classification
                    print(f"=== Running Single-Label Classification (RandomForest) for {groupName} ===")
                    model = RandomForest("RandomForest", groupData.get_embeddings(), groupData.get_type())
                    # Run training in thread pool
                    await loop.run_in_executor(None, lambda: model.train(groupData))
                    group["model"] = model
                
                return group
            
            # Run all training tasks concurrently
            tasks = [train_group(group) for group in groups]
            trained_groups = await asyncio.gather(*tasks)
            
            # Update groups in data dictionary
            data["groups"] = trained_groups
            
            # Publish event with trained models
            return await self.eventBus.publishAsync(Config.EVENT_MODEL_TRAINED, data)
        except Exception as e:
            print(f"Error training models: {e}")
            traceback.print_exc()
            return None
    
    def makePredictions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions with trained models and publish predictions_made event
        
        Args:
            data (dict): Dictionary containing groups with trained models
            
        Returns:
            dict: Updated data dictionary with predictions
        """
        groups = data["groups"]
        
        print("Making predictions...")
        try:
            for group in groups:
                groupName = group["name"]
                groupData = group["data"]
                model = group["model"]
                
                print(f"\nMaking predictions for group: {groupName}")
                
                # Make predictions
                model.predict(groupData.X_test)
            
            # Publish event with predictions
            return self.eventBus.publish(Config.EVENT_PREDICTIONS_MADE, data)
        except Exception as e:
            print(f"Error making predictions: {e}")
            traceback.print_exc()
            return None
    
    async def makePredictionsAsync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously make predictions with trained models
        
        Uses parallel processing to make predictions for different groups simultaneously.
        
        Args:
            data (dict): Dictionary containing groups with trained models
            
        Returns:
            dict: Updated data dictionary with predictions
        """
        groups = data["groups"]
        
        print("Making predictions asynchronously...")
        try:
            loop = asyncio.get_event_loop()
            
            # Prepare prediction tasks for each group
            async def predict_group(group):
                groupName = group["name"]
                groupData = group["data"]
                model = group["model"]
                
                print(f"\nMaking predictions for group: {groupName}")
                
                # Run prediction in thread pool
                await loop.run_in_executor(None, lambda: model.predict(groupData.X_test))
                
                return group
            
            # Run all prediction tasks concurrently
            tasks = [predict_group(group) for group in groups]
            predicted_groups = await asyncio.gather(*tasks)
            
            # Update groups in data dictionary
            data["groups"] = predicted_groups
            
            # Publish event with predictions
            return await self.eventBus.publishAsync(Config.EVENT_PREDICTIONS_MADE, data)
        except Exception as e:
            print(f"Error making predictions: {e}")
            traceback.print_exc()
            return None
    
    def evaluateModel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model performance and publish evaluation_completed event
        
        Args:
            data (dict): Dictionary containing groups with predictions
            
        Returns:
            dict: Data dictionary with evaluation results
        """
        groups = data["groups"]
        
        print("Evaluating models...")
        try:
            for group in groups:
                groupName = group["name"]
                groupData = group["data"]
                model = group["model"]
                
                print(f"\nEvaluation for group: {groupName}")
                
                # Print evaluation results
                model.print_results(groupData)
            
            # Store evaluation results if needed
            # ...
            
            # Publish event with evaluation results
            return self.eventBus.publish(Config.EVENT_EVALUATION_COMPLETED, data)
        except Exception as e:
            print(f"Error evaluating models: {e}")
            traceback.print_exc()
            return None
    
    async def evaluateModelAsync(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously evaluate model performance
        
        Uses parallel processing to evaluate different groups simultaneously.
        
        Args:
            data (dict): Dictionary containing groups with predictions
            
        Returns:
            dict: Data dictionary with evaluation results
        """
        groups = data["groups"]
        
        print("Evaluating models asynchronously...")
        try:
            loop = asyncio.get_event_loop()
            
            # Prepare evaluation tasks for each group
            async def evaluate_group(group):
                groupName = group["name"]
                groupData = group["data"]
                model = group["model"]
                
                print(f"\nEvaluation for group: {groupName}")
                
                # Run evaluation in thread pool to avoid blocking
                await loop.run_in_executor(None, lambda: model.print_results(groupData))
                
                return group
            
            # Run all evaluation tasks concurrently
            tasks = [evaluate_group(group) for group in groups]
            evaluated_groups = await asyncio.gather(*tasks)
            
            # Update groups in data dictionary
            data["groups"] = evaluated_groups
            
            # Publish event with evaluation results
            return await self.eventBus.publishAsync(Config.EVENT_EVALUATION_COMPLETED, data)
        except Exception as e:
            print(f"Error evaluating models: {e}")
            traceback.print_exc()
            return None
    
    def runPipeline(self, approach: Optional[str] = None):
        """
        Run the ML pipeline synchronously with the specified classification approach
        
        Args:
            approach (str, optional): Classification approach to use (CHAINED_OUTPUT or HIERARCHICAL)
        """
        # Set classification approach if specified
        if approach:
            Config.CLASSIFICATION_APPROACH = approach
            
        print(f"\n\n===== RUNNING {Config.CLASSIFICATION_APPROACH} APPROACH =====\n")
        
        # Start the pipeline by loading data
        # This will trigger the event chain through the subscribers
        self.loadData()
    
    async def runPipelineAsync(self, approach: Optional[str] = None):
        """
        Run the ML pipeline asynchronously with the specified classification approach
        
        Args:
            approach (str, optional): Classification approach to use (CHAINED_OUTPUT or HIERARCHICAL)
        """
        # Set classification approach if specified
        if approach:
            Config.CLASSIFICATION_APPROACH = approach
            
        print(f"\n\n===== RUNNING ASYNC {Config.CLASSIFICATION_APPROACH} APPROACH =====\n")
        
        # Start the pipeline by loading data
        # This will trigger the event chain through the subscribers
        await self.loadDataAsync() 