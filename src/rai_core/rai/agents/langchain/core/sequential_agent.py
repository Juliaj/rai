# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial
from typing import List, Optional, TypedDict, Dict, Any

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rai.agents.langchain.core.tool_runner import ToolRunner


class SequentialState(TypedDict):
    messages: List[BaseMessage]
    current_step: int
    required_tools: List[str]
    completed_steps: List[str]


def sequential_tools_condition(state: SequentialState) -> Dict[str, Any]:
    """
    Custom conditional routing that enforces tool usage in a specific sequence.
    
    This function checks if the LLM should use tools based on the current step
    and whether the required tool for that step has been called.
    """
    messages = state["messages"]
    current_step = state.get("current_step", 0)
    required_tools = state.get("required_tools", [])
    completed_steps = state.get("completed_steps", [])
    
    # If no more steps required, end
    if current_step >= len(required_tools):
        return {"end": None}
    
    # Check if the last message is from AI and contains tool calls
    if not messages or not isinstance(messages[-1], AIMessage):
        return {"thinker": None}
    
    last_message = messages[-1]
    if not last_message.tool_calls:
        return {"thinker": None}
    
    # Check if the required tool for current step was called
    required_tool = required_tools[current_step]
    tool_called = any(call["name"] == required_tool for call in last_message.tool_calls)
    
    if tool_called:
        # Move to next step
        state["current_step"] = current_step + 1
        state["completed_steps"].append(required_tool)
        return {"tools": None}
    else:
        # Wrong tool called, go back to thinker
        return {"thinker": None}


def agent(
    llm: BaseChatModel,
    logger: logging.Logger,
    system_prompt: str | SystemMessage,
    required_tools: List[str],
    state: SequentialState,
):
    logger.info(f"Running thinker - step {state.get('current_step', 0)}")
    
    # If there are no messages, do nothing
    if len(state["messages"]) == 0:
        return state
    
    # Insert system message if not already present
    if not isinstance(state["messages"][0], SystemMessage):
        system_msg = (
            SystemMessage(content=system_prompt)
            if isinstance(system_prompt, str)
            else system_prompt
        )
        state["messages"].insert(0, system_msg)
    
    # Add step-specific instructions to the system prompt
    current_step = state.get("current_step", 0)
    if current_step < len(required_tools):
        step_instruction = f"\n\nCURRENT STEP {current_step + 1}: You MUST use the tool '{required_tools[current_step]}' next. Do not use any other tools."
        if isinstance(state["messages"][0], SystemMessage):
            state["messages"][0].content += step_instruction
    
    ai_msg = llm.invoke(state["messages"])
    state["messages"].append(ai_msg)
    return state


def create_sequential_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    required_tools: List[str],
    system_prompt: str | SystemMessage,
    logger: Optional[logging.Logger] = None,
    debug: bool = False,
) -> CompiledStateGraph:
    """
    Create an agent that forces tools to be used in a specific sequence.
    
    Parameters
    ----------
    llm : BaseChatModel
        The language model to use
    tools : List[BaseTool]
        All available tools
    required_tools : List[str]
        List of tool names in the order they must be used
    system_prompt : str | SystemMessage
        The system prompt for the agent
    logger : Optional[logging.Logger]
        Logger instance
    debug : bool
        Whether to enable debug mode
        
    Returns
    -------
    CompiledStateGraph
        The compiled agent workflow
    """
    _logger = logger or logging.getLogger(__name__)
    
    _logger.info("Creating sequential agent")
    
    # Validate that all required tools exist
    tool_names = {tool.name for tool in tools}
    missing_tools = [name for name in required_tools if name not in tool_names]
    if missing_tools:
        raise ValueError(f"Required tools not found: {missing_tools}")
    
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolRunner(tools=tools, logger=_logger)
    
    workflow = StateGraph(SequentialState)
    workflow.add_node("tools", tool_node)
    workflow.add_node(
        "thinker", 
        partial(agent, llm_with_tools, _logger, system_prompt, required_tools)
    )
    
    workflow.add_edge(START, "thinker")
    workflow.add_edge("tools", "thinker")
    
    # Use custom conditional routing
    workflow.add_conditional_edges(
        "thinker",
        sequential_tools_condition,
    )
    
    app = workflow.compile(debug=debug)
    _logger.info("Sequential agent created")
    return app 