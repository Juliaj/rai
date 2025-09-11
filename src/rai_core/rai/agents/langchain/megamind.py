# Copyright (C) 2025 Robotec.AI
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


from functools import partial
from typing import (
    Annotated,
    List,
    Optional,
)

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from rai.agents.langchain.core.tool_runner import SubAgentToolRunner
from rai.messages import (
    HumanMultimodalMessage,
)


class StepSuccess(BaseModel):
    """Output of success attacher"""

    success: bool = Field(description="Whether the task was completed successfully")
    explanation: str = Field(description="Explanation of what happened")


class State(MessagesState):
    """State of the megamind agent"""
    original_task: str
    # steps done by the executor agent, the results of the steps
    steps_done: List[str]
    step: Optional[str]
    step_success: StepSuccess
    # messages from the user ?
    messages: List[BaseMessage]
    # executor context for execution conversation
    executor_messages: List[BaseMessage]


def llm_node(
    llm: BaseChatModel,
    system_prompt: Optional[str],
    state: State,
) -> State:
    """Process messages using the LLM - returns the agent's response."""
    messages = state["executor_messages"].copy()
    if not state["step"]:
        raise ValueError("Step should be defined at this point")
    if system_prompt:
        messages.insert(0, HumanMessage(state["step"]))
        messages.insert(0, SystemMessage(content=system_prompt))
    
    # context: messages, system_prompt, step
    ai_msg = llm.invoke(messages)

    # Only update executor context - don't pollute megamind context
    state["executor_messages"].append(ai_msg)
    state["messages"].append(ai_msg)
    return state

# structured output node analyze the outcome of a step. 
# It takes in the executor_messages and calls llm to generate a structured output - analysis of the step
# then it updates the state (step_success, steps_done) with the analysis
def structured_output_node(
    llm: BaseChatModel,
    task_planning_prompt: str,
    state: State,
) -> State:
    """Analyze the conversation and return structured output including the status of the task."""

    analyzer = llm.with_structured_output(StepSuccess)

    analysis = analyzer.invoke(
        [
            SystemMessage(
                content=f"""
Analyze if this task was completed successfully:

Task: {state["step"]}

{task_planning_prompt}

Below you have messages of agent doing the task:"""
            ),
            *state["executor_messages"],
        ]
    )

    state["step_success"] = StepSuccess(
        success=analysis.success, explanation=analysis.explanation
    )

    state["steps_done"].append(f"{state['step_success'].explanation}")
    return state

# conditional edge to decide whether to continue with tools or return structured output
# if the last message has tool calls, continue to tools
# otherwise, return structured output
# used by megamind agent to decide whether to continue with tools or return structured output
def should_continue_or_structure(state: State) -> str:
    """Decide whether to continue with tools or return structured output."""
    last_message = state["executor_messages"][-1]

    # If AI message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, return structured output
    return "structured_output"

# executor workflow
def create_react_structured_agent(
    llm: BaseChatModel,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    task_planning_prompt: str = f"""
Determine success and provide brief explanation of the task completion.""",
) -> CompiledStateGraph:
    """Create a react agent that returns structured output."""

    graph = StateGraph(State)
    graph.add_edge(START, "llm")

    if tools:
        tool_runner = SubAgentToolRunner(tools)
        graph.add_node("tools", tool_runner)

        # bind tools to the llm
        bound_llm = llm.bind_tools(tools)
        graph.add_node("llm", partial(llm_node, bound_llm, system_prompt))

        graph.add_node("structured_output", partial(structured_output_node, llm, task_planning_prompt))

        graph.add_conditional_edges(
            "llm",
            should_continue_or_structure,
            {"tools": "tools", "structured_output": "structured_output"},
        )
        graph.add_edge("tools", "llm")
        graph.add_edge("structured_output", END)
    else:
        graph.add_node("llm", partial(llm_node, llm, system_prompt))
        graph.add_node("structured_output", partial(structured_output_node, llm, task_planning_prompt))
        graph.add_edge("llm", "structured_output")
        graph.add_edge("structured_output", END)

    return graph.compile()


def create_megamind(
    manipulation_tools: List[BaseTool],
    megamind_llm: BaseChatModel,
    executor_llm: BaseChatModel,
    system_prompt: str,
    task_planning_prompt: str = f"""
Determine success and provide brief explanation of the task completion.""",
) -> CompiledStateGraph:

    def create_handoff_tool(*, agent_name: str, description: str | None = None):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            task_instruction: str,  # The specific task for the agent
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            return Command(
                goto=agent_name,
                # Send only the task message to the specialist agent, not the full history
                update={"step": task_instruction, "executor_messages": []},
                graph=Command.PARENT,
            )

        return handoff_tool


    executor_system_prompt = """You are a specialist robot agent with access to manipulation and navigation tools.
You can handle navigation tasks in space using provided tools.

After performing navigation action, always check your current position to ensure success

You can also handle object manipulation tasks including picking up and droping objects using provided tools.

Ask the VLM for objects detection and positions before performing any manipulation action.
If VLM doesn't see objects that are objectives of the task, return this information, without proceeding


"""

    # Create specialist agents
    executor_agent = create_react_structured_agent(
        llm=executor_llm,
        system_prompt=executor_system_prompt,
        tools=manipulation_tools,
        task_planning_prompt=task_planning_prompt,
    )

    assign_to_executor_agent = create_handoff_tool(
        agent_name="executor",
        description="Assign task to a executor agent.",
    )

    megamind_system_prompt = """You analyze the given long tasks and make a plan to pass it to specialists to whom you will delegate tasks. 
    You also monitor the specialists and provide feedback to them."""

    system_prompt += "\n"
    system_prompt += megamind_system_prompt

    # create a langchain reactive agent
    megamind_agent = create_react_agent(
        megamind_llm,
        tools=[assign_to_executor_agent],
        prompt=system_prompt,
        name="megamind",
    )

    # megamind action: plan the next step and delegate it to the executor agent
    # populate the original task and init step and step_done for State
    def plan_step(state: State) -> State:
        """Initial planning step."""
        if "original_task" not in state:
            state["original_task"] = state["messages"][0].content[0]["text"]
        if "steps_done" not in state:
            state["steps_done"] = []
        if "step" not in state:
            state["step"] = None

        megamind_prompt = (
            f"You are given objective to complete: {state['original_task']}"
        )
        if state["steps_done"]:
            megamind_prompt += "\n\n"
            megamind_prompt += "Steps that were already done successfully. Once a slot is marked as COMPLETED, stop thinking about this slot.\n"
            steps_done = "\n".join(
                [f"{i + 1}. {step}" for i, step in enumerate(state["steps_done"])]
            )
            megamind_prompt += steps_done
            megamind_prompt += "\n"

        if state["step"]:
            if not state["step_success"]:
                raise ValueError("Step success should be specified at this point")

            megamind_prompt += "\nBased on that outcome and past steps come up with the next step and delegate it to selected agent."

        else:
            megamind_prompt += "\n"
            megamind_prompt += (
                "Come up with the fist step and delegate it to selected agent."
            )

        megamind_prompt += "\n\n"
        megamind_prompt += (
            "When you decide that the objective is completed return response to user."
        )
        messages = [
            HumanMultimodalMessage(content=megamind_prompt),
        ]
        # NOTE (jmatejcz) the response of megamind isnt appended to messages
        # as Command from handoff instantly transitions to next node
        megamind_agent.invoke({"messages": messages})
        return state

    megamind = (
        StateGraph(State)
        .add_node("megamind", plan_step)
        .add_node("executor", executor_agent)
        .add_edge(START, "megamind")
        # always return back to the supervisor
        .add_edge("executor", "megamind")
        .compile()
    )

    return megamind
