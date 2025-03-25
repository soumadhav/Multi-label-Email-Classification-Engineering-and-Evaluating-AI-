"""
Main entry point for ML Pipeline with Event-Driven Architecture
Supports both synchronous and asynchronous operation
"""

from Config import Config
from events.MLPipeline import MLPipeline
import traceback
import asyncio
import time
import argparse

async def run_async_pipeline():
    """
    Run the ML pipeline asynchronously
    Uses parallel processing for improved performance
    """
    try:
        # Create ML Pipeline with event-driven architecture (async mode)
        pipeline = MLPipeline(useAsync=True)
        
        # Run pipeline with Chained Multi-Output Classification (Design Decision 1)
        await pipeline.runPipelineAsync(Config.CHAINED_OUTPUT)
        
        # Run pipeline with Hierarchical Modeling (Design Decision 2)
        await pipeline.runPipelineAsync(Config.HIERARCHICAL)
        
    except Exception as e:
        print(f"Error running async ML pipeline: {e}")
        traceback.print_exc()

def run_sync_pipeline():
    """
    Run the ML pipeline synchronously
    Traditional sequential processing
    """
    try:
        # Create ML Pipeline with event-driven architecture (sync mode)
        pipeline = MLPipeline(useAsync=False)
        
        # Run pipeline with Chained Multi-Output Classification (Design Decision 1)
        pipeline.runPipeline(Config.CHAINED_OUTPUT)
        
        # Run pipeline with Hierarchical Modeling (Design Decision 2)
        pipeline.runPipeline(Config.HIERARCHICAL)
        
    except Exception as e:
        print(f"Error running sync ML pipeline: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    """
    Run the ML pipeline with both classification approaches,
    allowing the user to choose between async and sync modes
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run ML pipeline with event-driven architecture')
    parser.add_argument('--async', dest='use_async', action='store_true',
                        help='Run in asynchronous mode (default: synchronous)')
    args = parser.parse_args()
    
    # Measure execution time
    start_time = time.time()
    
    if args.use_async:
        print("Running in ASYNCHRONOUS mode")
        # Run asynchronously
        asyncio.run(run_async_pipeline())
    else:
        print("Running in SYNCHRONOUS mode")
        # Run synchronously
        run_sync_pipeline()
    
    # Print execution time
    end_time = time.time()
    print(f"\nExecution completed in {end_time - start_time:.2f} seconds")

