# Multi-label Email Classification System

## Project Overview

This project implements a machine learning system for multi-label email classification using an event-driven architecture. The system classifies emails into multiple categories across three hierarchical levels (Type 2, Type 3, and Type 4) using two different architectural approaches:

1. **Chained Multi-Output Classification**
2. **Hierarchical Modeling**

The solution satisfies the requirements for a modular, extensible AI project architecture using software engineering principles like separation of concerns, encapsulation, and abstraction.

## Directory Structure

```
Actvity 3 Full Solution/
├── .idea/                    # IDE-specific files
├── data/                     # Email datasets
├── events/                   # Event-driven architecture components
│   ├── __init__.py
│   ├── EventBus.py           # Central event management system
│   └── MLPipeline.py         # Pipeline orchestrator
├── model/                    # ML model implementations
│   ├── __init__.py
│   ├── base.py               # Abstract base class for models
│   ├── chained_model.py      # Implementation of Chained approach
│   ├── hierarchical_model.py # Implementation of Hierarchical approach
│   └── randomforest.py       # Basic RandomForest implementation
├── modelling/                # Data handling and transformation
│   ├── __init__.py
│   ├── data_model.py         # Encapsulates data for consistent access
│   └── modelling.py          # Handles model selection and prediction
├── __pycache__/              # Python bytecode cache
├── Config.py                 # Configuration settings
├── embeddings.py             # Text vectorization functions
├── main.py                   # Entry point for application
├── out.csv                   # Output results
├── preprocess.py             # Data preprocessing functions
└── README.md                 # Project documentation
```

## Core Architecture Principles

### 1. Separation of Concerns (SoC)

The code is organized by functionality, separating:
- Data preprocessing (preprocess.py)
- Text vectorization (embeddings.py)
- Data modeling (modelling/data_model.py)
- Model implementation (model/ directory)
- Application control (main.py)
- Event management (events/ directory)

This ensures that changes in one area (e.g., preprocessing) do not impact other areas (e.g., model training).

### 2. Data Encapsulation

The `Data` class in `modelling/data_model.py` encapsulates:
- Training data (X_train, y_train)
- Testing data (X_test, y_test)
- Multiple target variables (y2, y3, y4)
- Class metadata

This provides a consistent interface for all models to access the necessary data.

### 3. Abstraction

The `BaseModel` abstract class in `model/base.py` defines a uniform interface that all models must implement:
- `train()`: Train the model
- `predict()`: Make predictions
- `print_results()`: Evaluate and display results
- `data_transform()`: Transform data if needed

This ensures all models maintain a consistent interface regardless of their internal implementation details.

### 4. Event-Driven Architecture

The `EventBus` class provides an event-driven system that:
- Allows loose coupling between components
- Supports both synchronous and asynchronous processing
- Makes the system more extensible and maintainable

## Classification Approaches

### Approach 1: Chained Multi-Output Classification

Implemented in `model/chained_model.py`, this approach:

1. Creates a separate model for each level:
   - Level 1: Predicts Type 2
   - Level 2: Uses Type 2 predictions + features to predict Type 3
   - Level 3: Uses Type 2 + Type 3 predictions + features to predict Type 4

2. Chains the predictions by using them as additional features:
   ```
   Features → Type 2 Model → Type 2 Predictions
   [Features + Type 2 Predictions] → Type 3 Model → Type 3 Predictions
   [Features + Type 2 Predictions + Type 3 Predictions] → Type 4 Model → Type 4 Predictions
   ```

3. Advantages:
   - Captures dependencies between levels
   - Uses information from previous levels to improve predictions
   - Maintains a uniform model structure

### Approach 2: Hierarchical Modeling

Implemented in `model/hierarchical_model.py`, this approach:

1. Creates a tree of models:
   - One model for Type 2 classification
   - Separate models for Type 3 classification, one for each Type 2 class
   - Separate models for Type 4 classification, one for each Type 2 + Type 3 combination

2. Data flow:
   ```
   Features → Type 2 Model → Type 2 Predictions
   
   For each Type 2 class:
     Filtered Features → Type 3 Model → Type 3 Predictions
     
     For each Type 3 class:
       Filtered Features → Type 4 Model → Type 4 Predictions
   ```

3. Advantages:
   - Creates specialized models for each branch
   - Can capture more specific patterns within branches
   - Better performance when previous level predictions are correct

## Event Flow

The system uses these events to coordinate processing:

1. `dataLoaded`: Raw data has been loaded from files
2. `dataPreprocessed`: Data has been cleaned and processed
3. `embeddingsCreated`: Text data has been converted to vectors
4. `dataTransformed`: Data has been split into training/testing sets
5. `modelTrained`: Models have been trained
6. `predictionsMade`: Predictions have been generated
7. `evaluationCompleted`: Results have been evaluated

## Running the System

### Requirements

- Python 3.6+
- Required libraries:
  - scikit-learn
  - pandas
  - numpy
  - scipy

### Installation

```bash
# Clone the repository
git clone https://github.com/soumadhav/Engineering-and-Evaluating

# Navigate to project directory
cd "Actvity 3 Full Solution"

# Install required packages
pip install -r requirements.txt  # If provided
# Or manually
pip install scikit-learn pandas numpy scipy
```

### Running the Application

```bash
# Run in synchronous mode (default)
python main.py

# Run in asynchronous mode (better performance)
python main.py --async
```

## Results Analysis

### Chained Multi-Output Classification Results

**AppGallery & Games Group:**
- Type 2 accuracy: 72%
- Type 3 accuracy: 60%
- Type 4 accuracy: 68%
- Overall accuracy (all levels correct): 52%

**In-App Purchase Group:**
- Type 2 accuracy: 82.35%
- Type 3 accuracy: 76.47%
- Type 4 accuracy: 70.59%
- Overall accuracy (all levels correct): 70.59%

### Hierarchical Modeling Results

**AppGallery & Games Group:**
- Type 2 accuracy: 72%
- Type 3 accuracy given correct Type 2: 82.34%
- Type 4 accuracy given correct Type 2 and Type 3: 88.89%
- Overall accuracy (all levels correct): 56%

**In-App Purchase Group:**
- Type 2 accuracy: 82%
- Type 3 accuracy given correct Type 2: 93.45%
- Type 4 accuracy given correct Type 2 and Type 3: 85.31%
- Overall accuracy (all levels correct): 70.59%

### Comparison of Approaches

1. **Performance**:
   - Hierarchical approach shows better performance for Type 3 and Type 4 when previous levels are correctly predicted
   - Hierarchical approach slightly outperforms chained approach for overall accuracy on AppGallery & Games (56% vs 52%)
   - Both approaches achieve identical overall accuracy on In-App Purchase (70.59%)

2. **Training Complexity**:
   - Chained approach: 3 models (one per level)
   - Hierarchical approach: 1 + N + N*M models (Type 2 classes + combinations)

3. **Prediction Time**:
   - Chained approach: More efficient for prediction
   - Hierarchical approach: May require more time due to multiple models

4. **Adaptability**:
   - Chained approach: Better for new combinations of classes
   - Hierarchical approach: Better for specific patterns within established branches

## File Descriptions

### Main Files

- **main.py**: Entry point for the application. Parses command-line arguments and runs the ML pipeline in either synchronous or asynchronous mode.

- **Config.py**: Stores configuration constants used throughout the application, such as event types, classification approaches, and column names.

- **preprocess.py**: Contains functions for data loading, cleaning, and preprocessing. Handles text normalization, deduplication, and noise removal.

- **embeddings.py**: Implements text vectorization using TF-IDF to convert text data into numerical features.

### Event System

- **events/EventBus.py**: Implements the event bus pattern for managing event subscriptions and publishing. Supports both synchronous and asynchronous operation.

- **events/MLPipeline.py**: Orchestrates the entire ML workflow using events. Defines handlers for each stage of processing and coordinates the flow.

### Models

- **model/base.py**: Abstract base class defining the interface that all model implementations must follow.

- **model/randomforest.py**: Basic RandomForest classifier implementation that follows the BaseModel interface.

- **model/chained_model.py**: Implements the Chained Multi-Output Classification approach for multi-label classification.

- **model/hierarchical_model.py**: Implements the Hierarchical Modeling approach for multi-label classification.

### Data Handling

- **modelling/data_model.py**: Defines the Data class that encapsulates training and testing data with accessor methods.

- **modelling/modelling.py**: Implements additional modeling functionality for data transformation and prediction.

## Additional Notes

- The system handles class imbalance using balanced class weights in RandomForest models
- Empty strings and NaN values are replaced with 'unknown' to ensure consistent processing
- Groups with too few samples (<3) are skipped to avoid overfitting
- The event-driven architecture allows for easy extension to add new processing steps or models