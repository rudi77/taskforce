How do multi-step agents work?
The ReAct framework (Yao et al., 2022) is currently the main approach to building agents.

The name is based on the concatenation of two words, “Reason” and “Act.” Indeed, agents following this architecture will solve their task in as many steps as needed, each step consisting of a Reasoning step, then an Action step where it formulates tool calls that will bring it closer to solving the task at hand.

All agents in smolagents are based on singular MultiStepAgent class, which is an abstraction of ReAct framework.

On a basic level, this class performs actions on a cycle of following steps, where existing variables and knowledge is incorporated into the agent logs like below:

Initialization: the system prompt is stored in a SystemPromptStep, and the user query is logged into a TaskStep .

While loop (ReAct loop):

Use agent.write_inner_memory_from_logs() to write the agent logs into a list of LLM-readable chat messages.
Send these messages to a Model object to get its completion. Parse the completion to get the action (a JSON blob for ToolCallingAgent, a code snippet for CodeAgent).
Execute the action and logs result into memory (an ActionStep).
At the end of each step, we run all callback functions defined in agent.step_callbacks .
Optionally, when planning is activated, a plan can be periodically revised and stored in a PlanningStep . This includes feeding facts about the task at hand to the memory.

For a CodeAgent, it looks like the figure below.


Here is a video overview of how that works:


Framework of a React Agent

We implement two versions of agents:

CodeAgent is the preferred type of agent: it generates its tool calls as blobs of code.
ToolCallingAgent generates tool calls as a JSON in its output, as is commonly done in agentic frameworks. We incorporate this option because it can be useful in some narrow cases where you can do fine with only one tool call per step: for instance, for web browsing, you need to wait after each action on the page to monitor how the page changes.
We also provide an option to run agents in one-shot: just pass single_step=True when launching the agent, like agent.run(your_task, single_step=True)
---

# CodeAgent Example Code:

```python
from typing import Optional

from smolagents import HfApiModel, LiteLLMModel, TransformersModel, tool
from smolagents.agents import CodeAgent, ToolCallingAgent


# Choose which inference type to use!

available_inferences = ["hf_api", "transformers", "ollama", "litellm"]
chosen_inference = "transformers"

print(f"Chose model: '{chosen_inference}'")

if chosen_inference == "hf_api":
    model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct")

elif chosen_inference == "transformers":
    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto", max_new_tokens=1000)

elif chosen_inference == "ollama":
    model = LiteLLMModel(
        model_id="ollama_chat/llama3.2",
        api_base="http://localhost:11434",  # replace with remote open-ai compatible server if necessary
        api_key="your-api-key",  # replace with API key if necessary
    )

elif chosen_inference == "litellm":
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-latest'
    model = LiteLLMModel(model_id="gpt-4o")


@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


agent = ToolCallingAgent(tools=[get_weather], model=model)

print("ToolCallingAgent:", agent.run("What's the weather like in Paris?"))

agent = CodeAgent(tools=[get_weather], model=model)

print("ToolCallingAgent:", agent.run("What's the weather like in Paris?"))
```

Another example:

```python
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    inspect,
    text,
)


engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# create city SQL table
table_name = "receipts"
receipts = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("customer_name", String(16), primary_key=True),
    Column("price", Float),
    Column("tip", Float),
)
metadata_obj.create_all(engine)

rows = [
    {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
    {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
    {"receipt_id": 3, "customer_name": "Woodrow Wilson", "price": 53.43, "tip": 5.43},
    {"receipt_id": 4, "customer_name": "Margaret James", "price": 21.11, "tip": 1.00},
]
for row in rows:
    stmt = insert(receipts).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

inspector = inspect(engine)
columns_info = [(col["name"], col["type"]) for col in inspector.get_columns("receipts")]

table_description = "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])
print(table_description)

from smolagents import tool


@tool
def sql_engine(query: str) -> str:
    """
    Allows you to perform SQL queries on the table. Returns a string representation of the result.
    The table is named 'receipts'. Its description is as follows:
        Columns:
        - receipt_id: INTEGER
        - customer_name: VARCHAR(16)
        - price: FLOAT
        - tip: FLOAT

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    with engine.connect() as con:
        rows = con.execute(text(query))
        for row in rows:
            output += "\n" + str(row)
    return output


from smolagents import CodeAgent, HfApiModel


agent = CodeAgent(
    tools=[sql_engine],
    model=HfApiModel("meta-llama/Meta-Llama-3.1-8B-Instruct"),
)
agent.run("Can you give me the name of the client who got the most expensive receipt?")
```

smolagents agent implementation is in __agents.py__

__agents.py__

#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .e2b_executor import E2BExecutor
from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonInterpreter,
    fix_final_answer_code,
)
from .models import (
    ChatMessage,
    MessageRole,
)
from .monitoring import Monitor
from .prompts import (
    CODE_SYSTEM_PROMPT,
    MANAGED_AGENT_PROMPT,
    PLAN_UPDATE_FINAL_PLAN_REDACTION,
    SYSTEM_PROMPT_FACTS,
    SYSTEM_PROMPT_FACTS_UPDATE,
    SYSTEM_PROMPT_PLAN,
    SYSTEM_PROMPT_PLAN_UPDATE,
    TOOL_CALLING_SYSTEM_PROMPT,
    USER_PROMPT_FACTS_UPDATE,
    USER_PROMPT_PLAN,
    USER_PROMPT_PLAN_UPDATE,
)
from .tools import (
    DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
    Tool,
    get_tool_description_with_args,
)
from .types import AgentAudio, AgentImage, handle_agent_output_types
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentLogger,
    AgentMaxStepsError,
    AgentParsingError,
    LogLevel,
    parse_code_blobs,
    parse_json_tool_call,
    truncate_content,
)


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str


class AgentStepLog:
    pass


@dataclass
class ActionStep(AgentStepLog):
    agent_memory: List[Dict[str, str]] | None = None
    tool_calls: List[ToolCall] | None = None
    start_time: float | None = None
    end_time: float | None = None
    step_number: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    llm_output: str | None = None
    observations: str | None = None
    observations_images: List[str] | None = None
    action_output: Any = None


@dataclass
class PlanningStep(AgentStepLog):
    plan: str
    facts: str


@dataclass
class TaskStep(AgentStepLog):
    task: str
    task_images: List[str] | None = None


@dataclass
class SystemPromptStep(AgentStepLog):
    system_prompt: str


def get_tool_descriptions(tools: Dict[str, Tool], tool_description_template: str) -> str:
    return "\n".join([get_tool_description_with_args(tool, tool_description_template) for tool in tools.values()])


def format_prompt_with_tools(tools: Dict[str, Tool], prompt_template: str, tool_description_template: str) -> str:
    tool_descriptions = get_tool_descriptions(tools, tool_description_template)
    prompt = prompt_template.replace("{{tool_descriptions}}", tool_descriptions)
    if "{{tool_names}}" in prompt:
        prompt = prompt.replace(
            "{{tool_names}}",
            ", ".join([f"'{tool.name}'" for tool in tools.values()]),
        )
    return prompt


def show_agents_descriptions(managed_agents: Dict):
    managed_agents_descriptions = """
You can also give requests to team members.
Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'request', a long string explaining your request.
Given that this team member is a real human, you should be very verbose in your request.
Here is a list of the team members that you can call:"""
    for agent in managed_agents.values():
        managed_agents_descriptions += f"\n- {agent.name}: {agent.description}"
    return managed_agents_descriptions


def format_prompt_with_managed_agents_descriptions(
    prompt_template,
    managed_agents,
    agent_descriptions_placeholder: Optional[str] = None,
) -> str:
    if agent_descriptions_placeholder is None:
        agent_descriptions_placeholder = "{{managed_agents_descriptions}}"
    if agent_descriptions_placeholder not in prompt_template:
        raise ValueError(
            f"Provided prompt template does not contain the managed agents descriptions placeholder '{agent_descriptions_placeholder}'"
        )
    if len(managed_agents.keys()) > 0:
        return prompt_template.replace(agent_descriptions_placeholder, show_agents_descriptions(managed_agents))
    else:
        return prompt_template.replace(agent_descriptions_placeholder, "")


YELLOW_HEX = "#d4b702"


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        system_prompt (`str`, *optional*): System prompt that will be used to generate the agent's actions.
        tool_description_template (`str`, *optional*): Template used to describe the tools in the system prompt.
        max_steps (`int`, default `6`): Maximum number of steps the agent can take to solve the task.
        tool_parser (`Callable`, *optional*): Function used to parse the tool calls from the LLM output.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`int`, default `1`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = None,
        max_steps: int = 6,
        tool_parser: Optional[Callable] = None,
        add_base_tools: bool = False,
        verbosity_level: int = 1,
        grammar: Optional[Dict[str, str]] = None,
        managed_agents: Optional[List] = None,
        step_callbacks: Optional[List[Callable]] = None,
        planning_interval: Optional[int] = None,
    ):
        if system_prompt is None:
            system_prompt = CODE_SYSTEM_PROMPT
        if tool_parser is None:
            tool_parser = parse_json_tool_call
        self.agent_name = self.__class__.__name__
        self.model = model
        self.system_prompt_template = system_prompt
        self.tool_description_template = (
            tool_description_template if tool_description_template else DEFAULT_TOOL_DESCRIPTION_TEMPLATE
        )
        self.max_steps = max_steps
        self.tool_parser = tool_parser
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state = {}

        self.managed_agents = {}
        if managed_agents is not None:
            self.managed_agents = {agent.name: agent for agent in managed_agents}

        for tool in tools:
            assert isinstance(tool, Tool), f"This element is not of class Tool: {str(tool)}"
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            for tool_name, tool_class in TOOL_MAPPING.items():
                if tool_name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent":
                    self.tools[tool_name] = tool_class()
        self.tools["final_answer"] = FinalAnswerTool()

        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.logs = []
        self.task = None
        self.logger = AgentLogger(level=verbosity_level)
        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)

    def initialize_system_prompt(self):
        self.system_prompt = format_prompt_with_tools(
            self.tools,
            self.system_prompt_template,
            self.tool_description_template,
        )
        self.system_prompt = format_prompt_with_managed_agents_descriptions(self.system_prompt, self.managed_agents)

        return self.system_prompt

    def write_inner_memory_from_logs(self, summary_mode: bool = False) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the logs into a series of messages
        that can be used as input to the LLM.

        Args:
            summary_mode (`bool`): Whether to write a summary of the logs or the full logs.
        """
        memory = []
        for i, step_log in enumerate(self.logs):
            if isinstance(step_log, SystemPromptStep):
                if not summary_mode:
                    thought_message = {
                        "role": MessageRole.SYSTEM,
                        "content": [{"type": "text", "text": step_log.system_prompt.strip()}],
                    }
                    memory.append(thought_message)

            elif isinstance(step_log, PlanningStep):
                thought_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": "[FACTS LIST]:\n" + step_log.facts.strip(),
                }
                memory.append(thought_message)

                if not summary_mode:
                    thought_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": "[PLAN]:\n" + step_log.plan.strip(),
                    }
                    memory.append(thought_message)

            elif isinstance(step_log, TaskStep):
                task_message = {
                    "role": MessageRole.USER,
                    "content": [{"type": "text", "text": f"New task:\n{step_log.task}"}],
                }
                if step_log.task_images:
                    for image in step_log.task_images:
                        task_message["content"].append({"type": "image", "image": image})
                memory.append(task_message)

            elif isinstance(step_log, ActionStep):
                if step_log.llm_output is not None and not summary_mode:
                    thought_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": [{"type": "text", "text": step_log.llm_output.strip()}],
                    }
                    memory.append(thought_message)
                if step_log.tool_calls is not None:
                    tool_call_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": [
                            {
                                "type": "text",
                                "text": str(
                                    [
                                        {
                                            "id": tool_call.id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_call.name,
                                                "arguments": tool_call.arguments,
                                            },
                                        }
                                        for tool_call in step_log.tool_calls
                                    ]
                                ),
                            }
                        ],
                    }
                    memory.append(tool_call_message)
                if step_log.error is not None:
                    error_message = {
                        "role": MessageRole.ASSISTANT,
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Error:\n"
                                    + str(step_log.error)
                                    + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                                ),
                            }
                        ],
                    }
                    memory.append(error_message)
                if step_log.observations is not None:
                    if step_log.tool_calls:
                        tool_call_reference = f"Call id: {(step_log.tool_calls[0].id if getattr(step_log.tool_calls[0], 'id') else 'call_0')}\n"
                    else:
                        tool_call_reference = ""
                    text_observations = f"Observation:\n{step_log.observations}"
                    tool_response_message = {
                        "role": MessageRole.TOOL_RESPONSE,
                        "content": [{"type": "text", "text": tool_call_reference + text_observations}],
                    }
                    memory.append(tool_response_message)
                if step_log.observations_images:
                    thought_message_image = {
                        "role": MessageRole.USER,
                        "content": [{"type": "text", "text": "Here are the observed images:"}]
                        + [
                            {
                                "type": "image",
                                "image": image,
                            }
                            for image in step_log.observations_images
                        ],
                    }
                    memory.append(thought_message_image)

        return memory

    def get_succinct_logs(self):
        return [{key: value for key, value in log.items() if key != "agent_memory"} for log in self.logs]

    def extract_action(self, llm_output: str, split_token: str) -> Tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            llm_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = llm_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{llm_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: Optional[list[str]]) -> str:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`, *optional*): Paths to image(s).

        Returns:
            `str`: Final answer to the task.
        """
        if images:
            self.input_messages[0]["content"] = [
                {
                    "type": "text",
                    "text": "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
                }
            ]
            self.input_messages[0]["content"].append({"type": "image"})
            self.input_messages += self.write_inner_memory_from_logs()[1:]
            self.input_messages += [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on the above, please provide an answer to the following user request:\n{task}",
                        }
                    ],
                }
            ]
        else:
            self.input_messages[0]["content"] = [
                {
                    "type": "text",
                    "text": "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:",
                }
            ]
            self.input_messages += self.write_inner_memory_from_logs()[1:]
            self.input_messages += [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on the above, please provide an answer to the following user request:\n{task}",
                        }
                    ],
                }
            ]
        try:
            return self.model(self.input_messages).content
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            raise AgentExecutionError(error_msg, self.logger)

        try:
            if isinstance(arguments, str):
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(arguments)
                else:
                    observation = available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(**arguments)
                else:
                    observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                error_msg = f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                raise AgentExecutionError(error_msg, self.logger)
            return observation
        except Exception as e:
            if tool_name in self.tools:
                tool_description = get_tool_description_with_args(available_tools[tool_name])
                error_msg = (
                    f"Error in tool call execution: {e}\nYou should only use this tool with a correct input.\n"
                    f"As a reminder, this tool's description is the following:\n{tool_description}"
                )
                raise AgentExecutionError(error_msg, self.logger)
            elif tool_name in self.managed_agents:
                error_msg = (
                    f"Error in calling team member: {e}\nYou should only ask this team member with a correct request.\n"
                    f"As a reminder, this team member's description is the following:\n{available_tools[tool_name]}"
                )
                raise AgentExecutionError(error_msg, self.logger)

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        single_step: bool = False,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            single_step (`bool`): Whether to run the agent in one-shot fashion.
            images (`list[str]`, *optional*): Paths to image(s).
            additional_args (`dict`): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """

        self.task = task
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.initialize_system_prompt()
        system_prompt_step = SystemPromptStep(system_prompt=self.system_prompt)

        if reset:
            self.logs = []
            self.logs.append(system_prompt_step)
            self.monitor.reset()
        else:
            if len(self.logs) > 0:
                self.logs[0] = system_prompt_step
            else:
                self.logs.append(system_prompt_step)

        self.logger.log(
            Panel(
                f"\n[bold]{self.task.strip()}\n",
                title="[bold]New run",
                subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=LogLevel.INFO,
        )

        self.logs.append(TaskStep(task=self.task, task_images=images))
        if single_step:
            step_start_time = time.time()
            step_log = ActionStep(start_time=step_start_time, observations_images=images)
            step_log.end_time = time.time()
            step_log.duration = step_log.end_time - step_start_time

            # Run the agent's step
            result = self.step(step_log)
            return result

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task, images=images)
        # Outputs are returned only at the end as a string. We only look at the last step
        return deque(self._run(task=self.task, images=images), maxlen=1)[0]

    def _run(self, task: str, images: List[str] | None = None) -> Generator[str, None, None]:
        """
        Run the agent in streaming mode and returns a generator of all the steps.

        Args:
            task (`str`): Task to perform.
            images (`list[str]`): Paths to image(s).
        """
        final_answer = None
        self.step_number = 0
        while final_answer is None and self.step_number < self.max_steps:
            step_start_time = time.time()
            step_log = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
            )
            try:
                if self.planning_interval is not None and self.step_number % self.planning_interval == 0:
                    self.planning_step(
                        task,
                        is_first_step=(self.step_number == 0),
                        step=self.step_number,
                    )
                self.logger.log(
                    Rule(
                        f"[bold]Step {self.step_number}",
                        characters="━",
                        style=YELLOW_HEX,
                    ),
                    level=LogLevel.INFO,
                )

                # Run one step!
                final_answer = self.step(step_log)
            except AgentError as e:
                step_log.error = e
            finally:
                step_log.end_time = time.time()
                step_log.duration = step_log.end_time - step_start_time
                self.logs.append(step_log)
                for callback in self.step_callbacks:
                    # For compatibility with old callbacks that don't take the agent as an argument
                    if len(inspect.signature(callback).parameters) == 1:
                        callback(step_log)
                    else:
                        callback(step_log=step_log, agent=self)
                self.step_number += 1
                yield step_log

        if final_answer is None and self.step_number == self.max_steps:
            error_message = "Reached max steps."
            final_step_log = ActionStep(
                step_number=self.step_number, error=AgentMaxStepsError(error_message, self.logger)
            )
            self.logs.append(final_step_log)
            final_answer = self.provide_final_answer(task, images)
            self.logger.log(Text(f"Final answer: {final_answer}"), level=LogLevel.INFO)
            final_step_log.action_output = final_answer
            final_step_log.end_time = time.time()
            final_step_log.duration = step_log.end_time - step_start_time
            for callback in self.step_callbacks:
                # For compatibility with old callbacks that don't take the agent as an argument
                if len(inspect.signature(callback).parameters) == 1:
                    callback(final_step_log)
                else:
                    callback(step_log=final_step_log, agent=self)
            yield final_step_log

        yield handle_agent_output_types(final_answer)

    def planning_step(self, task, is_first_step: bool, step: int) -> None:
        """
        Used periodically by the agent to plan the next steps to reach the objective.

        Args:
            task (`str`): Task to perform.
            is_first_step (`bool`): If this step is not the first one, the plan should be an update over a previous plan.
            step (`int`): The number of the current step, used as an indication for the LLM.
        """
        if is_first_step:
            message_prompt_facts = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_FACTS,
            }
            message_prompt_task = {
                "role": MessageRole.USER,
                "content": f"""Here is the task:
```
{task}
```
Now begin!""",
            }

            answer_facts = self.model([message_prompt_facts, message_prompt_task]).content

            message_system_prompt_plan = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_PLAN,
            }
            message_user_prompt_plan = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_PLAN.format(
                    task=task,
                    tool_descriptions=get_tool_descriptions(self.tools, self.tool_description_template),
                    managed_agents_descriptions=(show_agents_descriptions(self.managed_agents)),
                    answer_facts=answer_facts,
                ),
            }
            answer_plan = self.model(
                [message_system_prompt_plan, message_user_prompt_plan],
                stop_sequences=["<end_plan>"],
            ).content

            final_plan_redaction = f"""Here is the plan of action that I will follow to solve the task:
```
{answer_plan}
```"""
            final_facts_redaction = f"""Here are the facts that I know so far:
```
{answer_facts}
```""".strip()
            self.logs.append(PlanningStep(plan=final_plan_redaction, facts=final_facts_redaction))
            self.logger.log(
                Rule("[bold]Initial plan", style="orange"),
                Text(final_plan_redaction),
                level=LogLevel.INFO,
            )
        else:  # update plan
            agent_memory = self.write_inner_memory_from_logs(
                summary_mode=False
            )  # This will not log the plan but will log facts

            # Redact updated facts
            facts_update_system_prompt = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_FACTS_UPDATE,
            }
            facts_update_message = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_FACTS_UPDATE,
            }
            facts_update = self.model([facts_update_system_prompt] + agent_memory + [facts_update_message]).content

            # Redact updated plan
            plan_update_message = {
                "role": MessageRole.SYSTEM,
                "content": SYSTEM_PROMPT_PLAN_UPDATE.format(task=task),
            }
            plan_update_message_user = {
                "role": MessageRole.USER,
                "content": USER_PROMPT_PLAN_UPDATE.format(
                    task=task,
                    tool_descriptions=get_tool_descriptions(self.tools, self.tool_description_template),
                    managed_agents_descriptions=(show_agents_descriptions(self.managed_agents)),
                    facts_update=facts_update,
                    remaining_steps=(self.max_steps - step),
                ),
            }
            plan_update = self.model(
                [plan_update_message] + agent_memory + [plan_update_message_user],
                stop_sequences=["<end_plan>"],
            ).content

            # Log final facts and plan
            final_plan_redaction = PLAN_UPDATE_FINAL_PLAN_REDACTION.format(task=task, plan_update=plan_update)
            final_facts_redaction = f"""Here is the updated list of the facts that I know:
```
{facts_update}
```"""
            self.logs.append(PlanningStep(plan=final_plan_redaction, facts=final_facts_redaction))
            self.logger.log(
                Rule("[bold]Updated plan", style="orange"),
                Text(final_plan_redaction),
                level=LogLevel.INFO,
            )


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        system_prompt (`str`, *optional*): System prompt that will be used to generate the agent's actions.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = TOOL_CALLING_SYSTEM_PROMPT
        super().__init__(
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            planning_interval=planning_interval,
            **kwargs,
        )

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.input_messages = agent_memory

        # Add new step in logs
        log_entry.agent_memory = agent_memory.copy()

        try:
            model_message = self.model(
                self.input_messages,
                tools_to_call_from=list(self.tools.values()),
                stop_sequences=["Observation:"],
            )
            if model_message.tool_calls is None or len(model_message.tool_calls) == 0:
                raise Exception("Model did not call any tools. Call `final_answer` tool to return a final answer.")
            tool_call = model_message.tool_calls[0]
            tool_name, tool_call_id = tool_call.function.name, tool_call.id
            tool_arguments = tool_call.function.arguments

        except Exception as e:
            raise AgentGenerationError(f"Error in generating tool call with model:\n{e}", self.logger) from e

        log_entry.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]

        # Execute
        self.logger.log(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            level=LogLevel.INFO,
        )
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            if (
                isinstance(answer, str) and answer in self.state.keys()
            ):  # if the answer is a state variable, return the value
                final_answer = self.state[answer]
                self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                final_answer = answer
                self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )

            log_entry.action_output = final_answer
            return final_answer
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments)
            observation_type = type(observation)
            if observation_type in [AgentImage, AgentAudio]:
                if observation_type == AgentImage:
                    observation_name = "image.png"
                elif observation_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: observation naming could allow for different names of same type

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            else:
                updated_information = str(observation).strip()
            self.logger.log(
                f"Observations: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            log_entry.observations = updated_information
            return None


class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        system_prompt (`str`, *optional*): System prompt that will be used to generate the agent's actions.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        use_e2b_executor (`bool`, default `False`): Whether to use the E2B executor for remote code execution.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        system_prompt: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        use_e2b_executor: bool = False,
        max_print_outputs_length: Optional[int] = None,
        **kwargs,
    ):
        if system_prompt is None:
            system_prompt = CODE_SYSTEM_PROMPT

        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        if "{{authorized_imports}}" not in system_prompt:
            raise ValueError("Tag '{{authorized_imports}}' should be provided in the prompt.")
        super().__init__(
            tools=tools,
            model=model,
            system_prompt=system_prompt,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )

        if use_e2b_executor and len(self.managed_agents) > 0:
            raise Exception(
                f"You passed both {use_e2b_executor=} and some managed agents. Managed agents is not yet supported with remote code execution."
            )

        all_tools = {**self.tools, **self.managed_agents}
        if use_e2b_executor:
            self.python_executor = E2BExecutor(
                self.additional_authorized_imports,
                list(all_tools.values()),
                self.logger,
            )
        else:
            self.python_executor = LocalPythonInterpreter(
                self.additional_authorized_imports,
                all_tools,
                max_print_outputs_length=max_print_outputs_length,
            )

    def initialize_system_prompt(self):
        super().initialize_system_prompt()
        self.system_prompt = self.system_prompt.replace(
            "{{authorized_imports}}",
            (
                "You can import from any package you want."
                if "*" in self.authorized_imports
                else str(self.authorized_imports)
            ),
        )
        return self.system_prompt

    def step(self, log_entry: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.input_messages = agent_memory.copy()

        # Add new step in logs
        log_entry.agent_memory = agent_memory.copy()
        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            llm_output = self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:"],
                **additional_args,
            ).content
            log_entry.llm_output = llm_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        self.logger.log(
            Group(
                Rule(
                    "[italic]Output message of the LLM:",
                    align="left",
                    style="orange",
                ),
                Syntax(
                    llm_output,
                    lexer="markdown",
                    theme="github-dark",
                    word_wrap=True,
                ),
            ),
            level=LogLevel.DEBUG,
        )

        # Parse
        try:
            code_action = fix_final_answer_code(parse_code_blobs(llm_output))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        log_entry.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.logs)}",
            )
        ]

        # Execute
        self.logger.log(
            Panel(
                Syntax(
                    code_action,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]Executing this code:",
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=LogLevel.INFO,
        )
        observation = ""
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(
                code_action,
                self.state,
            )
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation += "Execution logs:\n" + execution_logs
        except Exception as e:
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        log_entry.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        log_entry.action_output = output
        return output if is_final_answer else None


class ManagedAgent:
    """
    ManagedAgent class that manages an agent and provides additional prompting and run summaries.

    Args:
        agent (`object`): The agent to be managed.
        name (`str`): The name of the managed agent.
        description (`str`): A description of the managed agent.
        additional_prompting (`Optional[str]`, *optional*): Additional prompting for the managed agent. Defaults to None.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary after the agent completes its task. Defaults to False.
        managed_agent_prompt (`Optional[str]`, *optional*): Custom prompt for the managed agent. Defaults to None.

    """

    def __init__(
        self,
        agent,
        name,
        description,
        additional_prompting: Optional[str] = None,
        provide_run_summary: bool = False,
        managed_agent_prompt: Optional[str] = None,
    ):
        self.agent = agent
        self.name = name
        self.description = description
        self.additional_prompting = additional_prompting
        self.provide_run_summary = provide_run_summary
        self.managed_agent_prompt = managed_agent_prompt if managed_agent_prompt else MANAGED_AGENT_PROMPT

    def write_full_task(self, task):
        """Adds additional prompting for the managed agent, like 'add more detail in your answer'."""
        full_task = self.managed_agent_prompt.format(name=self.name, task=task)
        if self.additional_prompting:
            full_task = full_task.replace("\n{additional_prompting}", self.additional_prompting).strip()
        else:
            full_task = full_task.replace("\n{additional_prompting}", "").strip()
        return full_task

    def __call__(self, request, **kwargs):
        full_task = self.write_full_task(request)
        output = self.agent.run(full_task, **kwargs)
        if self.provide_run_summary:
            answer = f"Here is the final answer from your managed agent '{self.name}':\n"
            answer += str(output)
            answer += f"\n\nFor more detail, find below a summary of this agent's work:\nSUMMARY OF WORK FROM AGENT '{self.name}':\n"
            for message in self.agent.write_inner_memory_from_logs(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += f"\nEND OF SUMMARY OF WORK FROM AGENT '{self.name}'."
            return answer
        else:
            return output


__all__ = [
    "ManagedAgent",
    "MultiStepAgent",
    "CodeAgent",
    "ToolCallingAgent",
]