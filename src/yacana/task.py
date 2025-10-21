import copy
import logging
import uuid
from typing import List, Type, Callable, Dict
from pydantic import BaseModel

from .generic_agent import GenericAgent, GenericMessage
from .exceptions import IllogicalConfiguration
from .history import History
from .logging_config import LoggerManager
from .tool import Tool
from .history import Message

LoggerManager.set_library_log_level("httpx", "WARNING")


class Task:
    """
    A class representing a task to be solved by an LLM agent.

    The best approach to use an LLM is to define a task with a clear objective.
    Then assign an LLM to this task so that it can try to solve it. You can also add Tools
    so that when the LLM starts solving it gets some tools relevant to the current task.
    This means that tools are not specific to an agent but to the task at hand. This allows
    more flexibility by producing less confusion to the LLM as it gets access to the tools
    it needs only when faced with a task that is related.

    Parameters
    ----------
    prompt : str
        The task to solve. It is the prompt given to the assigned LLM.
    agent : GenericAgent
        The agent assigned to this task.
    json_output : bool, optional
        If True, will force the LLM to answer as JSON. Defaults to False.
    structured_output : Type[BaseModel] | None, optional
        The expected structured output type for the task. If provided, the LLM's response
        will be validated against this type. Defaults to None.
    tools : List[Tool], optional
        A list of tools that the LLM will get access to when trying to solve this task.
        Defaults to an empty list.
    medias : List[str] | None, optional
        An optional list of paths pointing to images on the filesystem. Defaults to None.
    llm_stops_by_itself : bool, optional
        Only useful when the task is part of a GroupSolve(). Signals the assigned LLM
        that it will have to stop talking by its own means. Defaults to False.
    use_self_reflection : bool, optional
        Only useful when the task is part of a GroupSolve(). Allows keeping the self
        reflection process done by the LLM in the next GS iteration. Defaults to False.
    forget : bool, optional
        When True, the Agent won't remember this task after completion. Useful for
        routing purposes. Defaults to False.
    streaming_callback : Callable | None, optional
        Optional callback for streaming responses. Defaults to None.
    runtime_config : Dict | None, optional
        Optional runtime configuration for the task. Defaults to None.
    tags : List[str] | None, optional
        Optional list of tags that will be added to all message(s) corresponding to this task. Defaults to None.

    Attributes
    ----------
    prompt : str
        The task to solve. It is the prompt given to the assigned LLM.
    agent : GenericAgent
        The agent assigned to this task.
    json_output : bool
        If True, will force the LLM to answer as JSON.
    structured_output : Type[BaseModel] | None
        The expected structured output type for the task.
    tools : List[Tool]
        A list of tools that the LLM will get access to when trying to solve this task.
    medias : List[str] | None
        An optional list of paths pointing to images on the filesystem.
    llm_stops_by_itself : bool
        Only useful when the task is part of a GroupSolve(). Signals the assigned LLM
        that it will have to stop talking by its own means.
    use_self_reflection : bool
        Only useful when the task is part of a GroupSolve(). Allows keeping the self
        reflection process done by the LLM in the next GS iteration.
    forget : bool
        When True, the Agent won't remember this task after completion. Useful for
        routing purposes.
    streaming_callback : Callable | None
        Optional callback for streaming responses.
    runtime_config : Dict | None
        Optional runtime configuration for the task.
    tags : List[str] | None
        Optional list of tags that will be added to all message(s) corresponding to this task.

    Raises
    ------
    IllogicalConfiguration
        If both tools and structured_output are provided, or if both streaming_callback
        and structured_output are provided.
    """

    def __init__(self, prompt: str, agent: GenericAgent, json_output=False, structured_output: Type[BaseModel] | None = None, tools: List[Tool] = None,
                 medias: List[str] | None = None, llm_stops_by_itself: bool = False, use_self_reflection=False, forget=False, streaming_callback: Callable | None = None, runtime_config: Dict | None = None, tags: List[str] = None) -> None:
        self.prompt: str = prompt
        self.agent: GenericAgent = agent
        self.json_output: bool = json_output
        self.structured_output: Type[BaseModel] | None = structured_output
        self.tools: List[Tool] = tools if tools is not None else []
        self.llm_stops_by_itself: bool = llm_stops_by_itself
        self.use_self_reflection: bool = use_self_reflection
        self.forget: bool = forget
        self.medias: List[str] | None = medias
        self._uuid: str = str(uuid.uuid4())
        self.streaming_callback: Callable | None = streaming_callback
        self.runtime_config = runtime_config if runtime_config is not None else {}
        self.tags: List[str] = tags if tags is not None else []

        if len(self.tools) > 0 and self.structured_output is not None:
            raise IllogicalConfiguration("You can't have tools and structured_output at the same time. The tool output will be considered the LLM output hence not using the structured output.")

        if self.streaming_callback is not None and self.structured_output is not None:
            raise IllogicalConfiguration("You can't have streaming_callback and structured_output at the same time. Having incomplete JSON is useless.")

        self._check_tools_are_same_type()
        self._check_tool_names_are_unique()

        # Only used when @forget is True
        self._initial_history: History | None = None

    @property
    def uuid(self) -> str:
        """
        Get the unique identifier for this task.

        Returns
        -------
        str
            A unique task identifier.
        """
        return self._uuid

    def _check_tools_are_same_type(self):
        """
        All tools must be of the same type. We can't mix tools because they are not proposed in the same way to the LLM.
        """
        if len(self.tools) > 0:
            first_tool_type = self.tools[0].tool_type
            for tool in self.tools:
                if tool.tool_type != first_tool_type:
                    raise IllogicalConfiguration("All tools must be of the same type. Mixing tool types is not allowed. Use ToolType.YACANA or ToolType.OPENAI to specify the type of tool execution you want to use.")

    def _check_tool_names_are_unique(self):
        """
        Check that all local tools have unique names and that MCP tools do not conflict with local tool names.
        """
        local_tool_names: List[str] = [tool.tool_name for tool in self.tools if not tool.is_mcp]

        if len(set(local_tool_names)) != len(local_tool_names):
            raise IllogicalConfiguration("All local tools must have unique names. Found duplicates in the task's tool list.")

        unique_mcp_tools: List[Tool] = []
        mcp_tools: List[Tool] = [tool for tool in self.tools if tool.is_mcp]

        for mcp_tool in mcp_tools:
            if mcp_tool.tool_name not in local_tool_names:
                unique_mcp_tools.append(mcp_tool)
            else:
                logging.warning(
                    "Tool '%s' is a MCP tool but its name '%s' is already used by a tool defined at Task "
                    "level. Local task tools take precedence upon remote MCP tools for security reasons. You should either change your local tool name or use the .forget_tool() method from the Mcp class. For now the remote tool will no be available.",
                    mcp_tool.tool_name,
                    mcp_tool.tool_name
                )

        self.tools = [tool for tool in self.tools if not tool.is_mcp] + unique_mcp_tools

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the list of tools available for this task.

        Parameters
        ----------
        tool : Tool
            The tool to add to the task's tool list.
        """
        self.tools.append(tool)

    def _save_history_state(self) -> None:
        """Save the current state of the agent's history."""
        if self.forget:
            self._initial_history = copy.deepcopy(self.agent.history)

    def _restore_history_state(self) -> None:
        """Restore the agent's history to its initial state."""
        if self.forget and self._initial_history is not None:
            self.agent.history = self._initial_history

    def solve(self) -> Message:
        """
        Execute the task using the assigned LLM agent.

        This method will call the assigned LLM to perform inference on the task's prompt.
        If tools are available, the LLM may use them, potentially making multiple calls
        to the inference server.

        Returns
        -------
        GenericMessage | None
            The last message from the LLM.

        Raises
        ------
        MaxToolErrorIter
            If tool errors exceed the limit.
        """
        self._save_history_state()
        answer: GenericMessage = self.agent._interact(self.prompt, self.tools, self.json_output, self.structured_output, self.medias, self.streaming_callback, self.runtime_config, self.tags)
        self._restore_history_state()
        return answer

