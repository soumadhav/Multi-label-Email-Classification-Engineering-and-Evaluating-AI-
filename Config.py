class Config:
    """
    Configuration class for ML pipeline settings.
    Contains constants used throughout the application.
    """
    # Input Columns - Used for data extraction
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns for multi-label classification
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'
    
    # Multi-label classification approaches
    CHAINED_OUTPUT = "chained"
    HIERARCHICAL = "hierarchical"
    
    # Default classification approach
    CLASSIFICATION_APPROACH = CHAINED_OUTPUT
    
    # Event types for event-driven architecture
    EVENT_DATA_LOADED = "dataLoaded"
    EVENT_DATA_PREPROCESSED = "dataPreprocessed" 
    EVENT_EMBEDDINGS_CREATED = "embeddingsCreated"
    EVENT_DATA_TRANSFORMED = "dataTransformed"
    EVENT_MODEL_TRAINED = "modelTrained"
    EVENT_PREDICTIONS_MADE = "predictionsMade"
    EVENT_EVALUATION_COMPLETED = "evaluationCompleted"