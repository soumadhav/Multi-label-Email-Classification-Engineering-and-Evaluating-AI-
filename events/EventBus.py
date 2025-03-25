"""
Event Bus Module for Event-Driven Architecture
Manages event subscriptions and publishing throughout the ML pipeline
Supports both synchronous and asynchronous event handling
"""
import asyncio
from typing import Dict, List, Callable, Any, Union, Awaitable

class EventBus:
    """
    Core component of event-driven architecture that facilitates communication 
    between publishers and subscribers. 
    
    Maintains a registry of event types and their subscribers, and routes
    events to appropriate handlers. Supports both synchronous and asynchronous
    event handling for improved performance.
    """
    
    def __init__(self):
        """
        Initialize the event bus with empty subscriber dictionaries for
        both synchronous and asynchronous subscribers
        """
        self.syncSubscribers = {}
        self.asyncSubscribers = {}
        
    def subscribe(self, eventType: str, callback: Callable):
        """
        Register a synchronous callback function for a specific event type
        
        Args:
            eventType (str): The event type to subscribe to
            callback (function): The function to call when the event occurs
        """
        if eventType not in self.syncSubscribers:
            self.syncSubscribers[eventType] = []
        self.syncSubscribers[eventType].append(callback)
    
    def subscribeAsync(self, eventType: str, callback: Callable):
        """
        Register an asynchronous callback function for a specific event type
        
        Args:
            eventType (str): The event type to subscribe to
            callback (function): The async function to call when the event occurs
        """
        if eventType not in self.asyncSubscribers:
            self.asyncSubscribers[eventType] = []
        self.asyncSubscribers[eventType].append(callback)
        
    def publish(self, eventType: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish an event to all synchronous subscribers
        
        Args:
            eventType (str): The type of event being published
            data (dict): The data associated with the event
            
        Returns:
            The data that was published (potentially modified by subscribers)
        """
        print(f"EVENT: {eventType}")
        if eventType in self.syncSubscribers:
            for callback in self.syncSubscribers[eventType]:
                data = callback(data)
        return data
    
    async def publishAsync(self, eventType: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish an event to all subscribers asynchronously
        
        This method handles both synchronous and asynchronous subscribers.
        Synchronous subscribers are run in a thread pool to avoid blocking.
        
        Args:
            eventType (str): The type of event being published
            data (dict): The data associated with the event
            
        Returns:
            The data that was published (potentially modified by subscribers)
        """
        print(f"ASYNC EVENT: {eventType}")
        
        # Handle synchronous subscribers in a non-blocking way
        if eventType in self.syncSubscribers:
            loop = asyncio.get_event_loop()
            for callback in self.syncSubscribers[eventType]:
                # Run synchronous callbacks in a thread pool
                data = await loop.run_in_executor(None, lambda: callback(data))
        
        # Handle asynchronous subscribers
        if eventType in self.asyncSubscribers:
            for callback in self.asyncSubscribers[eventType]:
                data = await callback(data)
                
        return data 