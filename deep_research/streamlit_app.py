import streamlit as st
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import Counter, defaultdict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the deep research components
from deep_research_graph import DeepResearchWorkflow, DeepResearchState

# Page configuration
st.set_page_config(
    page_title="Deep Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'workflow' not in st.session_state:
    st.session_state.workflow = None
if 'events_log' not in st.session_state:
    st.session_state.events_log = []
if 'research_complete' not in st.session_state:
    st.session_state.research_complete = False
if 'final_result' not in st.session_state:
    st.session_state.final_result = None

def setup_environment():
    """Setup environment variables and check if everything is configured"""
    st.sidebar.header("🔧 Configuration")
    
    # Check for required environment variables
    required_vars = ['OPENROUTER_API_KEY', 'FIRECRAWL_API_KEY', 'SERP_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.sidebar.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.sidebar.info("Please set these in your .env file or environment")
        return False
    
    st.sidebar.success("✅ All environment variables configured")
    return True

def create_workflow_settings():
    """Create workflow configuration settings"""
    st.sidebar.header("🔬 Research Settings")
    
    max_steps = st.sidebar.slider(
        "Maximum Research Steps",
        min_value=1,
        max_value=15,
        value=5,
        help="Maximum number of research steps the workflow will perform"
    )
    
    return max_steps

def format_event_for_display(event: Dict[str, Any]) -> Dict[str, Any]:
    """Format event data for better display"""
    formatted = {
        'timestamp': event['timestamp'].strftime('%H:%M:%S'),
        'node': event['node'],
        'details': {}
    }
    
    output = event['output']
    if isinstance(output, dict):
        # Extract relevant information from different node types
        if 'current_plan' in output and output['current_plan']:
            plan = output['current_plan']
            if hasattr(plan, 'steps'):
                formatted['details']['plan_steps'] = len(plan.steps)
                formatted['details']['steps_preview'] = plan.steps[:2]
        
        if 'step_count' in output:
            formatted['details']['steps_completed'] = output['step_count']
        
        if 'current_step' in output and output['current_step']:
            formatted['details']['current_step'] = output['current_step'][:100]
        
        if 'step_result' in output and output['step_result']:
            formatted['details']['result_length'] = len(str(output['step_result']))
        
        if 'final_answer' in output and output['final_answer']:
            formatted['details']['final_answer_length'] = len(output['final_answer'])
            formatted['details']['has_final_answer'] = True
        
        if 'error_message' in output and output['error_message']:
            formatted['details']['error'] = output['error_message']
    
    return formatted

def display_streaming_events():
    """Display streaming events in real-time"""
    st.subheader("📡 Live Event Stream")
    
    # Create container for events
    events_container = st.container()
    
    with events_container:
        if st.session_state.events_log:
            for i, event in enumerate(st.session_state.events_log):
                formatted_event = format_event_for_display(event)
                
                # Create columns for better layout
                col1, col2, col3 = st.columns([1, 2, 4])
                
                with col1:
                    st.write(f"**{formatted_event['timestamp']}**")
                
                with col2:
                    node_name = formatted_event['node']
                    if node_name == 'initial_planning':
                        st.write("🎯 **Planning**")
                    elif node_name == 'step_execution':
                        st.write("🔍 **Executing**")
                    elif node_name == 'replanning':
                        st.write("🔄 **Replanning**")
                    elif node_name == 'summarize':
                        st.write("📝 **Summarizing**")
                    else:
                        st.write(f"📋 **{node_name}**")
                
                with col3:
                    details = formatted_event['details']
                    if 'plan_steps' in details:
                        st.write(f"Generated plan with {details['plan_steps']} steps")
                    elif 'steps_completed' in details:
                        st.write(f"Completed step {details['steps_completed']}")
                    elif 'current_step' in details:
                        st.write(f"Executing: {details['current_step']}...")
                    elif 'result_length' in details:
                        st.write(f"Retrieved {details['result_length']} characters of data")
                    elif 'has_final_answer' in details:
                        st.write(f"Final answer ready ({details['final_answer_length']} characters)")
                    elif 'error' in details:
                        st.error(f"Error: {details['error']}")
                    else:
                        st.write("Processing...")
                
                st.divider()
        else:
            st.info("No events yet. Start a research session to see live updates.")

def run_research_workflow(objective: str, max_steps: int):
    """Run the research workflow with streaming updates"""
    
    # Initialize workflow
    workflow = DeepResearchWorkflow(max_steps=max_steps)
    st.session_state.workflow = workflow
    
    # Create initial state
    initial_state = DeepResearchState(
        objective=objective,
        max_steps=max_steps
    )
    
    # Clear previous events
    st.session_state.events_log = []
    st.session_state.research_complete = False
    st.session_state.final_result = None
    
    # Add initial state to events log
    st.session_state.events_log.append({
        'timestamp': datetime.now(),
        'node': 'initial_state',
        'output': initial_state.model_dump()
    })
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Stream the workflow
        total_events = 0
        for event in workflow.workflow.stream(initial_state):
            total_events += 1
            
            # Get node name and output
            node_name = list(event.keys())[0]
            node_output = event[node_name]
            
            # Add to events log
            st.session_state.events_log.append({
                'timestamp': datetime.now(),
                'node': node_name,
                'output': node_output
            })
            
            # Update progress
            if isinstance(node_output, dict) and 'step_count' in node_output:
                progress = min(node_output['step_count'] / max_steps, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Step {node_output['step_count']}/{max_steps} completed")
            
            # Check if we have final answer
            if isinstance(node_output, dict) and 'final_answer' in node_output and node_output['final_answer']:
                st.session_state.final_result = node_output['final_answer']
                st.session_state.research_complete = True
                progress_bar.progress(1.0)
                status_text.text("Research complete!")
                break
            
            # Force UI update
            time.sleep(0.1)
        
        # Add final result to events log
        if st.session_state.final_result:
            st.session_state.events_log.append({
                'timestamp': datetime.now(),
                'node': 'final_result',
                'output': st.session_state.final_result
            })
    
    except Exception as e:
        st.error(f"Error during research: {str(e)}")
        st.session_state.events_log.append({
            'timestamp': datetime.now(),
            'node': 'error',
            'output': str(e)
        })

def display_final_results():
    """Display the final research results"""
    if st.session_state.final_result:
        st.subheader("📋 Final Research Results")
        
        # Display the final answer in a nice format
        st.markdown("### 🎯 Research Summary")
        st.markdown(st.session_state.final_result)
        
        # Add download button for results
        st.download_button(
            label="📥 Download Results",
            data=st.session_state.final_result,
            file_name=f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def create_performance_analysis():
    """Create performance analysis visualizations"""
    if not st.session_state.events_log or len(st.session_state.events_log) < 2:
        st.info("Not enough events for performance analysis.")
        return
    
    st.subheader("📊 Performance Analysis")
    
    # Analyze events
    events_df = pd.DataFrame([
        {
            'timestamp': event['timestamp'],
            'node': event['node'],
            'time_seconds': (event['timestamp'] - st.session_state.events_log[0]['timestamp']).total_seconds()
        }
        for event in st.session_state.events_log
    ])
    
    # Calculate time spent per node (duration between consecutive events)
    per_event_durations = []
    for i in range(1, len(st.session_state.events_log)):
        prev_time = st.session_state.events_log[i-1]['timestamp']
        curr_time = st.session_state.events_log[i]['timestamp']
        duration = (curr_time - prev_time).total_seconds()
        per_event_durations.append({
            'node': st.session_state.events_log[i]['node'],
            'duration': duration
        })
    
    # Aggregate total time spent per node
    node_time_spent = defaultdict(float)
    for entry in per_event_durations:
        node_time_spent[entry['node']] += entry['duration']
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart: time spent per node
        if node_time_spent:
            # Sort nodes by total time descending
            sorted_nodes_times = sorted(node_time_spent.items(), key=lambda x: x[1], reverse=True)
            nodes = [node for node, _ in sorted_nodes_times]
            times = [t for _, t in sorted_nodes_times]
            
            fig_bar = px.bar(
                x=times,
                y=nodes,
                orientation='h',
                title="Time Spent per Workflow Node",
                labels={'x': 'Time (seconds)', 'y': 'Workflow Node'}
            )
            
            # Add value labels on bars
            fig_bar.update_traces(
                texttemplate='%{x:.2f}s',
                textposition='outside'
            )
            
            fig_bar.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No timing data available for bar chart.")
    
    with col2:
        # Timeline chart
        fig_timeline = px.line(
            events_df,
            x=range(len(events_df)),
            y='time_seconds',
            title="Event Timeline",
            labels={'x': 'Event Sequence', 'y': 'Time (seconds)'}
        )
        fig_timeline.add_scatter(
            x=list(range(len(events_df))),
            y=events_df['time_seconds'],
            mode='markers+text',
            text=events_df['node'],
            textposition='top center',
            name='Events'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Performance metrics
    total_time = (st.session_state.events_log[-1]['timestamp'] - st.session_state.events_log[0]['timestamp']).total_seconds()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(st.session_state.events_log))
    with col2:
        st.metric("Total Time", f"{total_time:.2f}s")
    with col3:
        st.metric("Avg Time/Event", f"{total_time/len(st.session_state.events_log):.2f}s")
    with col4:
        research_steps = len([e for e in st.session_state.events_log if e['node'] == 'step_execution'])
        st.metric("Research Steps", research_steps)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔍 Deep Research Assistant")
    st.markdown("*Powered by LangGraph and advanced research workflows*")
    
    # Sidebar setup
    if not setup_environment():
        st.stop()
    
    max_steps = create_workflow_settings()
    
    # Main interface
    st.header("🎯 Research Objective")
    
    # Research input
    objective = st.text_area(
        "Enter your research objective:",
        placeholder="e.g., Write a comprehensive analysis of the latest AI developments in 2024, focusing on LLM capabilities and market trends.",
        height=100
    )
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_research = st.button("🚀 Start Research", disabled=not objective or st.session_state.research_complete)
    
    with col2:
        clear_results = st.button("🗑️ Clear Results")
    
    if clear_results:
        st.session_state.events_log = []
        st.session_state.research_complete = False
        st.session_state.final_result = None
        st.rerun()
    
    # Start research workflow
    if start_research and objective:
        st.info("🔄 Starting research workflow...")
        run_research_workflow(objective, max_steps)
        st.rerun()
    
    # Display streaming events
    if st.session_state.events_log:
        display_streaming_events()
    
    # Display final results
    if st.session_state.research_complete and st.session_state.final_result:
        display_final_results()
        
        # Performance analysis
        create_performance_analysis()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.markdown("""
    This Deep Research Assistant uses LangGraph to create a sophisticated 
    research workflow that can:
    
    - 📋 Generate research plans
    - 🔍 Execute research steps
    - 🔄 Adapt based on findings
    - 📝 Summarize results
    - 📊 Provide performance insights
    """)

if __name__ == "__main__":
    main() 