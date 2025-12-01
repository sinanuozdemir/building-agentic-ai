#!/usr/bin/env python3
"""
Celery app configuration and tasks for Generic Agent
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any

from celery import Celery
from celery.result import AsyncResult

from agent_builder import GenericAgent

# Celery configuration
celery_app = Celery(
    'agent_tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'amqp://agent:agent123@localhost:5672/%2Fagent'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'rpc://')
)

# Celery configuration settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # 4 minutes soft limit
    worker_prefetch_multiplier=1,
    result_expires=3600,  # Results expire after 1 hour
)

# Global agent instances cache for workers
agent_cache = {}

def get_agent(agent_config: str = None):
    """Get or create generic agent instance with specified config"""
    config_key = agent_config or "default"
    
    if config_key not in agent_cache:
        if agent_config:
            # Create agent with custom config
            agent_cache[config_key] = GenericAgent(config_path=agent_config)
        else:
            # Create agent with default config
            agent_cache[config_key] = GenericAgent()
    
    return agent_cache[config_key]

@celery_app.task(bind=True, name='agent_tasks.process_chat')
def process_chat_task(self, message: str, thread_id: str = None, agent_config: str = None) -> Dict[str, Any]:
    """
    Celery task to process chat messages with the generic agent
    
    Args:
        message: The user's message/question
        thread_id: Conversation thread ID for context
        agent_config: Optional path to agent configuration file
        
    Returns:
        Dict containing the agent's response and metadata
    """
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={
                'message': message[:100] + '...' if len(message) > 100 else message,
                'thread_id': thread_id,
                'agent_config': agent_config,
                'started_at': datetime.now().isoformat()
            }
        )
        
        # Get agent instance with specified config
        agent = get_agent(agent_config)
        
        # Create new event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize agent if needed
            if not agent.initialized:
                loop.run_until_complete(agent.initialize())
            
            # Process the chat message
            result = loop.run_until_complete(agent.chat(message, thread_id))
            
            # Add task metadata
            result.update({
                'task_id': self.request.id,
                'processed_at': datetime.now().isoformat(),
                'thread_id': thread_id,
                'agent_config': agent_config
            })
            
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        # Log the error and return failure state
        error_msg = str(e)
        print(f"❌ Error in process_chat_task: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'task_id': self.request.id,
            'timestamp': datetime.now().isoformat(),
            'thread_id': thread_id,
            'agent_config': agent_config
        }

@celery_app.task(bind=True, name='agent_tasks.get_tools')
def get_available_tools_task(self, agent_config: str = None) -> Dict[str, Any]:
    """
    Celery task to get available tools from the generic agent
    
    Args:
        agent_config: Optional path to agent configuration file
    
    Returns:
        Dict containing available tools list
    """
    try:
        self.update_state(
            state='PROCESSING',
            meta={
                'action': 'fetching_tools',
                'agent_config': agent_config,
                'started_at': datetime.now().isoformat()
            }
        )
        
        # Get agent instance with specified config
        agent = get_agent(agent_config)
        
        # Create new event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize agent if needed
            if not agent.initialized:
                loop.run_until_complete(agent.initialize())
            
            # Get available tools
            tools = loop.run_until_complete(agent.get_available_tools())
            
            return {
                'success': True,
                'tools': tools,
                'tools_count': len(tools),
                'task_id': self.request.id,
                'agent_config': agent_config,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in get_available_tools_task: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'task_id': self.request.id,
            'agent_config': agent_config,
            'timestamp': datetime.now().isoformat()
        }

@celery_app.task(bind=True, name='agent_tasks.get_conversation_history')
def get_conversation_history_task(self, thread_id: str, agent_config: str = None) -> Dict[str, Any]:
    """
    Celery task to get conversation history for a thread
    
    Args:
        thread_id: Conversation thread ID
        agent_config: Optional path to agent configuration file
        
    Returns:
        Dict containing conversation history
    """
    try:
        self.update_state(
            state='PROCESSING',
            meta={
                'action': 'fetching_history',
                'thread_id': thread_id,
                'agent_config': agent_config,
                'started_at': datetime.now().isoformat()
            }
        )
        
        # Get agent instance with specified config
        agent = get_agent(agent_config)
        
        # Create new event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize agent if needed
            if not agent.initialized:
                loop.run_until_complete(agent.initialize())
            
            # Get conversation history
            history = loop.run_until_complete(agent.get_conversation_history(thread_id))
            
            return {
                'success': True,
                'thread_id': thread_id,
                'history': history,
                'task_id': self.request.id,
                'agent_config': agent_config,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in get_conversation_history_task: {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'task_id': self.request.id,
            'thread_id': thread_id,
            'agent_config': agent_config,
            'timestamp': datetime.now().isoformat()
        }

def get_task_result(task_id: str) -> Dict[str, Any]:
    """
    Get result of a Celery task
    
    Args:
        task_id: The task ID to check
        
    Returns:
        Dict containing task status and result
    """
    try:
        # Get task result with explicit backend connection
        result = AsyncResult(task_id, app=celery_app)
        
        # Try to force a fresh lookup
        try:
            # This will force a backend lookup
            _ = result.result
        except Exception:
            pass
        
        print(f"🔍 Task {task_id} state: {result.state}")
        print(f"🔍 Task {task_id} ready: {result.ready()}")
        print(f"🔍 Task {task_id} successful: {result.successful()}")
        
        # Check if task is ready (completed, successful or failed)
        if result.ready():
            if result.successful():
                print(f"🔍 Task {task_id} result: {result.result}")
                response = {
                    'state': 'SUCCESS',
                    'result': result.result,
                    'task_id': task_id
                }
            else:
                print(f"🔍 Task {task_id} failed: {result.traceback}")
                response = {
                    'state': 'FAILURE',
                    'error': str(result.info),
                    'task_id': task_id
                }
        else:
            # Task is still processing
            if result.state == 'PROCESSING':
                response = {
                    'state': result.state,
                    'status': 'Task is being processed',
                    'meta': result.info,
                    'task_id': task_id
                }
            else:
                response = {
                    'state': 'PENDING',
                    'status': 'Task is waiting to be processed',
                    'task_id': task_id
                }
            
        return response
        
    except Exception as e:
        print(f"❌ Error getting task result for {task_id}: {e}")
        return {
            'state': 'ERROR',
            'error': f'Failed to get task result: {str(e)}',
            'task_id': task_id
        }

# Export the celery app
celery = celery_app 