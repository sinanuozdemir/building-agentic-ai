#!/usr/bin/env python3
"""
Flask app with Generic LangGraph React Agent and MCP integration
Now with Celery async task support
"""

import os
import asyncio
import json
import glob
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage

from agent_builder import GenericAgent
from celery_app import process_chat_task, get_available_tools_task, get_conversation_history_task, get_task_result

print(f'OPENROUTER_API_KEY: {os.getenv("OPENROUTER_API_KEY")}')

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'

# Global agent instance for synchronous endpoints
generic_agent = GenericAgent()

# Global agent instances cache
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

def discover_agent_configs():
    """Dynamically discover agent configuration files"""
    configs = {}
    
    # Get the directory where the Flask app is running
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all JSON files that could be agent configs
    config_patterns = [
        os.path.join(app_dir, 'agent_config*.json'),  # Original pattern
        os.path.join(app_dir, '*_config.json'),       # Files ending with _config.json
        os.path.join(app_dir, '*_agent.json'),        # Files ending with _agent.json
        os.path.join(app_dir, 'sdr*.json'),           # SDR specific configs
        os.path.join(app_dir, 'qualifying*.json')     # Qualifying specific configs
    ]
    
    config_files = []
    for pattern in config_patterns:
        config_files.extend(glob.glob(pattern))
    
    # Remove duplicates while preserving order
    config_files = list(dict.fromkeys(config_files))
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Extract filename for the key
            filename = os.path.basename(config_file)
            
            # Extract name and description from config
            name = config_data.get('name', filename.replace('.json', '').replace('_', ' ').title())
            description = config_data.get('description', 'No description available')
            
            configs[filename] = {
                'name': name,
                'description': description,
                'filename': filename
            }
            
        except Exception as e:
            print(f"Error loading config file {config_file}: {str(e)}")
            continue
    
    return configs

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/agent-configs')
def get_agent_configs():
    """Get available agent configurations"""
    try:
        configs = discover_agent_configs()
        return jsonify({
            "success": True,
            "configs": configs
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "initialized": generic_agent.initialized,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "flask": "running",
            "celery": "available",
            "rabbitmq": "connected"
        }
    })

# =============================================================================
# SYNCHRONOUS ENDPOINTS
# =============================================================================

@app.route('/api/initialize', methods=['POST'])
def initialize_agent():
    """Initialize the generic agent"""
    try:
        data = request.get_json() or {}
        agent_config = data.get('agent_config')
        
        # Initialize the agent with the specified config
        agent = get_agent(agent_config)
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        loop.run_until_complete(agent.initialize())
        tools = loop.run_until_complete(agent.get_available_tools())
        
        return jsonify({
            "success": True,
            "message": "Generic Agent initialized successfully with persistent memory",
            "tools_count": len(tools),
            "tools": tools
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with the generic agent (synchronous)"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data.get('message')
    conversation_id = data.get('conversation_id', data.get('thread_id', 'default'))
    agent_config = data.get('agent_config')
    
    print(f"🔍 Sync Chat - Conversation ID: {conversation_id}, Agent Config: {agent_config}")
    
    # Get the appropriate agent instance
    agent = get_agent(agent_config)
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(agent.chat(message, conversation_id))
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 500

@app.route('/api/tools')
def get_tools():
    """Get available tools (synchronous)"""
    try:
        agent_config = request.args.get('agent_config')
        agent = get_agent(agent_config)
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tools = loop.run_until_complete(agent.get_available_tools())
        return jsonify({
            "success": True,
            "tools": tools
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/conversations/<conversation_id>')
def get_conversation(conversation_id):
    """Get conversation history (synchronous)"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        history = loop.run_until_complete(generic_agent.get_conversation_history(conversation_id))
        
        return jsonify({
            "conversation_id": conversation_id,
            "history": history
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def clear_conversation(conversation_id):
    """Clear conversation history (synchronous)"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(generic_agent.clear_conversation(conversation_id))
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Conversation {conversation_id} cleared successfully"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to clear conversation"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =============================================================================
# ASYNCHRONOUS ENDPOINTS
# =============================================================================

@app.route('/api/async/chat', methods=['POST'])
def start_chat_task():
    """Start an async chat task and return task ID"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data.get('message')
    thread_id = data.get('thread_id', data.get('conversation_id', 'default'))
    agent_config = data.get('agent_config')  # Optional agent configuration
    
    try:
        # Start the Celery task with optional agent config
        task = process_chat_task.delay(message, thread_id, agent_config)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": "Chat task started",
            "thread_id": thread_id,
            "agent_config": agent_config,
            "status_url": f"/api/async/tasks/{task.id}",
            "timestamp": datetime.now().isoformat()
        }), 202  # 202 Accepted
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to start chat task: {str(e)}"
        }), 500

@app.route('/api/async/conversations/<thread_id>', methods=['POST'])
def start_conversation_history_task(thread_id):
    """Start an async task to get conversation history"""
    data = request.get_json() or {}
    agent_config = data.get('agent_config')  # Optional agent configuration
    
    try:
        # Start the Celery task with optional agent config
        task = get_conversation_history_task.delay(thread_id, agent_config)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": f"Conversation history task started for thread {thread_id}",
            "thread_id": thread_id,
            "agent_config": agent_config,
            "status_url": f"/api/async/tasks/{task.id}",
            "timestamp": datetime.now().isoformat()
        }), 202  # 202 Accepted
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to start conversation history task: {str(e)}"
        }), 500

@app.route('/api/async/tools', methods=['GET'])
def start_tools_task():
    """Start an async task to get available tools"""
    agent_config = request.args.get('agent_config')
    
    try:
        # Start the Celery task with optional agent config
        task = get_available_tools_task.delay(agent_config)
        
        return jsonify({
            "success": True,
            "task_id": task.id,
            "message": "Tools task started",
            "agent_config": agent_config,
            "status_url": f"/api/async/tasks/{task.id}",
            "timestamp": datetime.now().isoformat()
        }), 202  # 202 Accepted
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to start tools task: {str(e)}"
        }), 500

@app.route('/api/async/tasks/<task_id>')
def get_task_status(task_id):
    """Get the status and result of an async task"""
    try:
        result = get_task_result(task_id)
        
        # Determine HTTP status code based on task state
        if result['state'] in ['PENDING', 'PROCESSING']:
            status_code = 202  # Still processing
        elif result['state'] == 'SUCCESS':
            status_code = 200  # Complete
        else:
            status_code = 500  # Error
            
        return jsonify(result), status_code
        
    except Exception as e:
        return jsonify({
            "state": "ERROR",
            "error": f"Failed to get task status: {str(e)}",
            "task_id": task_id
        }), 500

@app.route('/api/async/tasks')
def list_task_endpoints():
    """List available async task endpoints"""
    return jsonify({
        "async_endpoints": {
            "start_chat": {
                "method": "POST",
                "url": "/api/async/chat",
                "description": "Start async chat with agent",
                "payload": {
                    "message": "Your question here",
                    "thread_id": "optional_thread_id"
                }
            },
            "start_conversation_history": {
                "method": "POST",
                "url": "/api/async/conversations/{thread_id}",
                "description": "Start async conversation history fetch"
            },
            "get_tools": {
                "method": "GET",
                "url": "/api/async/tools",
                "description": "Get available tools for specified agent config"
            },
            "get_task_status": {
                "method": "GET",
                "url": "/api/async/tasks/{task_id}",
                "description": "Get task status and result"
            }
        },
        "task_states": {
            "PENDING": "Task is waiting to be processed",
            "PROCESSING": "Task is currently being processed",
            "SUCCESS": "Task completed successfully",
            "FAILURE": "Task failed with error",
            "ERROR": "System error occurred"
        }
    })

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Agent Flask App with Celery support...")
    print("🔧 Make sure your services are running:")
    print("   docker-compose up -d")
    print("🌐 Synchronous API: http://localhost:5000")
    print("🌐 Asynchronous API: http://localhost:5000/api/async/")
    print("🐰 RabbitMQ Management: http://localhost:15672 (agent/agent123)")
    print("💾 Conversations are persisted using LangGraph's MemorySaver")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    ) 