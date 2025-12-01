from typing import Dict, Any, List, Tuple, Annotated, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
import os
import json

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Import our existing modules
from planning import Plan, planner, replanner
from step_executor import search_serp_tool, create_step_executor


class DeepResearchState(BaseModel):
    """State for deep research workflow"""
    
    # Input
    objective: str = Field(description="The research objective from the user")
    
    # Planning
    current_plan: Optional[Plan] = Field(default=None, description="Current research plan")
    
    # Execution tracking
    completed_steps: List[Tuple[str, str]] = Field(default_factory=list, description="List of (step, result) tuples")
    current_step: Optional[str] = Field(default=None, description="Current step being executed")
    step_result: Optional[str] = Field(default=None, description="Result of current step execution")
    
    # Progress tracking
    step_count: int = Field(default=0, description="Number of steps completed")
    max_steps: int = Field(default=5, description="Maximum number of research steps")
    
    # Summarization
    ready_to_summarize: bool = Field(default=False, description="Whether research is ready for final summarization")
    initial_response: Optional[str] = Field(default=None, description="Initial response from replanner before summarization")
    
    # Final output
    final_answer: Optional[str] = Field(default=None, description="Final research response")
    is_complete: bool = Field(default=False, description="Whether research is complete")
    
    # Metadata
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    error_message: Optional[str] = Field(default=None, description="Any error that occurred")


class DeepResearchWorkflow:
    """LangGraph workflow for deep research"""
    
    def __init__(self, max_steps: int = 5):
        """
        Initialize the deep research workflow
        
        Args:
            max_steps: Maximum number of research steps to perform
        """
        self.max_steps = max_steps
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for deep research"""
        
        def initial_planning(state: DeepResearchState) -> Dict[str, Any]:
            """Create initial research plan from user objective"""
            
            try:
                # Create messages for the planner
                messages = [HumanMessage(content=f"Objective: {state.objective}")]
                
                # Generate initial plan
                plan = planner.invoke({"messages": messages})
                
                # Add message to track plan
                # updated_messages = state.messages.copy()
                # updated_messages.append()
                
                return {
                    "current_plan": plan,
                    "messages": [*state.messages, AIMessage(content=f"Created research plan with {len(plan.steps)} steps.")]
                }
                
            except Exception as e:
                error_msg = f"Error in initial planning: {str(e)}"
                return {
                    "error_message": error_msg,
                    "is_complete": True
                }
        
        def step_execution(state: DeepResearchState) -> Dict[str, Any]:
            """Execute the current research step"""
            
            if not state.current_plan or not state.current_plan.steps:
                return {
                    "error_message": "No plan or steps available for execution",
                    "is_complete": True
                }
            
            # Get current step to execute
            current_step = state.current_plan.steps[0]  # Always take first remaining step
            step_num = state.step_count + 1
            
            try:
                completed_steps = state.completed_steps.copy()
                if completed_steps:
                    completed_summary = "\n".join(
                        [f"Step: {step}\nResult: {result['messages'][-1].content if isinstance(result, dict) and 'messages' in result else str(result)}" for step, result in completed_steps]
                    )
                    prompt = (
                        f"Given the objective {state.objective}, the current plan {state.current_plan}, "
                        f"the completed steps and their results so far:\n{completed_summary}\n"
                        f"execute the step {current_step} and return the result."
                    )
                else:
                    prompt = (
                        f"Given the objective {state.objective} and the current plan {state.current_plan}, "
                        f"execute the step {current_step} and return the result."
                    )
                prompt += "\n\n" + "Please return the result in a well thought out manner and include all sources of information found along the way so that the user can understand the research process. Basically recap every single thing you did and what you found with cited sources."

                agent_executor = create_step_executor("openai/gpt-4.1-mini")
                agent_response = agent_executor.invoke({"messages": [HumanMessage(content=prompt)]})

                # Extract the result from the agent response
                if "messages" in agent_response and agent_response["messages"]:
                    # Get the last AI message
                    last_message = agent_response["messages"][-1]
                    if hasattr(last_message, 'content'):
                        step_result = last_message.content
                    else:
                        step_result = str(last_message)
                else:
                    step_result = str(agent_response)
                
                # Add to completed steps
                completed_steps = state.completed_steps.copy()
                completed_steps.append((current_step, step_result))
                
                # Update messages
                updated_messages = state.messages.copy()
                updated_messages.append(HumanMessage(content=f"Execute: {current_step}"))
                updated_messages.append(AIMessage(content=f"ReAct agent completed research. Generated comprehensive analysis ({len(step_result)} chars)."))
                
                return {
                    "current_step": current_step,
                    "step_result": step_result,
                    "completed_steps": completed_steps,
                    "step_count": step_num,
                    "messages": updated_messages
                }
                
            except Exception as e:
                error_msg = f"Error executing step with ReAct agent: {str(e)}"
                # Fall back to direct search if agent fails
                try:
                    fallback_result = search_serp_tool.run(current_step)
                    completed_steps = state.completed_steps.copy()
                    completed_steps.append((current_step, f"Fallback search result: {fallback_result}"))
                    
                    return {
                        "current_step": current_step,
                        "step_result": fallback_result,
                        "completed_steps": completed_steps,
                        "step_count": step_num,
                        "error_message": f"Agent failed, used fallback: {error_msg}"
                    }
                except Exception as fallback_error:
                    # Complete failure
                    completed_steps = state.completed_steps.copy()
                    completed_steps.append((current_step, f"Error: {error_msg}"))
                    
                    return {
                        "current_step": current_step,
                        "step_result": error_msg,
                        "completed_steps": completed_steps,
                        "step_count": step_num,
                        "error_message": error_msg
                    }
        
        def replanning(state: DeepResearchState) -> Dict[str, Any]:
            """Update plan based on completed steps"""
            
            try:
                # Invoke replanner with current state
                updated_plan = replanner.invoke({
                    "input": state.objective,
                    "plan": state.current_plan,
                    "past_steps": state.completed_steps
                })
                
                updated_messages = state.messages.copy()
                
                if hasattr(updated_plan.action, 'steps'):
                    # More steps needed
                    new_plan = Plan(steps=updated_plan.action.steps)
                    
                    updated_messages.append(AIMessage(content=f"Plan updated. {len(new_plan.steps)} steps remaining."))
                    
                    return {
                        "current_plan": new_plan,
                        "messages": updated_messages,
                        "is_complete": False
                    }
                else:
                    # Research complete - route to summarize step
                    initial_response = updated_plan.action.response
                    
                    updated_messages.append(AIMessage(content="Research completed. Moving to final summarization."))
                    
                    return {
                        "initial_response": initial_response,
                        "is_complete": True,
                        "messages": updated_messages
                    }
                    
            except Exception as e:
                error_msg = f"Error in replanning: {str(e)}"
                return {
                    "error_message": error_msg,
                    "is_complete": True
                }
        
        def summarize(state: DeepResearchState) -> Dict[str, Any]:
            """Generate final comprehensive response using LLM"""
            
            try:
                # Create summarization prompt
                summarize_prompt = ChatPromptTemplate.from_template(
                    """You are an expert research analyst tasked with delivering a comprehensive final response.

Original Research Objective: {objective}

Research Steps Completed:
{completed_steps}

Initial Analysis: {initial_response}

Based on all the research conducted, please provide a well-structured, comprehensive final response that directly addresses the original objective. Your response should:
1. Synthesize findings from all research steps
2. Present information in a clear, organized manner
3. Directly answer the original objective
4. Be complete and actionable

Deliver your final response:"""
                )
                
                # Format completed steps for the prompt
                steps_summary = "\n".join([
                    f"Step {i+1}: {step}\nResult: {result}\n"
                    for i, (step, result) in enumerate(state.completed_steps)
                ])
                
                # Create LLM for summarization
                summarizer = summarize_prompt | ChatOpenAI(
                    model="deepseek/deepseek-r1-0528-qwen3-8b", temperature=0.3, base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
                )
                
                # Generate final response
                final_response = summarizer.invoke({
                    "objective": state.objective,
                    "completed_steps": steps_summary,
                    "initial_response": state.initial_response or "No initial response provided"
                })
                
                # Update messages
                updated_messages = state.messages.copy()
                updated_messages.append(AIMessage(content="Generated comprehensive final response."))
                updated_messages.append(AIMessage(content=final_response.content))
                
                return {
                    "final_answer": final_response.content,
                    "is_complete": True,
                    "messages": updated_messages
                }
                
            except Exception as e:
                error_msg = f"Error in summarization: {str(e)}"
                # Fallback to initial response if summarization fails
                return {
                    "final_answer": state.initial_response or "Error occurred during summarization",
                    "is_complete": True,
                    "error_message": error_msg
                }
        
        def should_continue(state: DeepResearchState) -> str:
            """Determine whether to continue research or end"""
            # Check if we've hit max steps
            if state.step_count >= state.max_steps:
                return "summarize"
            
            # Check if research is marked as complete
            if state.is_complete:
                return "summarize"
            
            # Check if we have an error
            if state.error_message:
                return "summarize"
            
            # Check if we have more steps to execute
            if state.current_plan and state.current_plan.steps:
                return "step_execution"
            
            # No more steps, finalize
            return "summarize"
        
        # Create the workflow graph
        workflow = StateGraph(DeepResearchState)
        
        # Add nodes
        workflow.add_node("initial_planning", initial_planning)
        workflow.add_node("step_execution", step_execution)  
        workflow.add_node("replanning", replanning)
        workflow.add_node("summarize", summarize)
        
        # Add edges
        workflow.add_edge(START, "initial_planning")
        workflow.add_edge("initial_planning", "step_execution")
        workflow.add_edge("step_execution", "replanning")
        workflow.add_edge("summarize", END)
        
        # Add conditional routing from replanning
        workflow.add_conditional_edges(
            "replanning",
            should_continue,
            {
                "step_execution": "step_execution",
                "summarize": "summarize",
                END: END
            }
        )
        
        return workflow.compile()
    
    def run_research(self, objective: str, max_steps: int = None) -> Dict[str, Any]:
        """Run the deep research workflow"""
        if max_steps is None:
            max_steps = self.max_steps
            
        initial_state = DeepResearchState(
            objective=objective,
            max_steps=max_steps
        )
        
        result = self.workflow.invoke(initial_state)
        
        return result
    