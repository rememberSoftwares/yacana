import copy
import uuid
from typing import List

from .agent import Agent, Message
from .exceptions import MaxToolErrorIter
from .history import History
from .logging_config import LoggerManager
from .tool import Tool

LoggerManager.set_library_log_level("httpx", "WARNING")


class Task:
    """The best approach to use an LLM is to define a task with a clear objective.
    Then assign a LLM to this task so that it can try to solve it. You can also add Tools so that when the LLM starts
    solving it gets some tools relevant to the current task. This means that tools are not specific to an agent but to
    the task at hand. This allows more flexibility by producing less confusion to the LLM as it gets access to the tools
    it needs only when faced with a task that is related.

    Attributes
    ----------
    task : str
        The task to solve. It is the prompt given to the assigned LLM
    agent : Agent
        The agent assigned to this task
    json_output : bool
        If True, will force the LLM to answer as JSON. Its using Ollama json mode for now. We shall see how to implement that on other inference backends. Either way you should ask for a JSON output in the task prompt.
    tools : List[Tool]
        A list of tools that the LLM will get access to when trying to solve this task. This means that tools are not bound to the LLM itself but to the task. This provides more flexibility and less confusion to the LLM as it gets access to the tools it needs in relation to the task at hand.
    raise_when_max_tool_error_iter : bool
        You should try/catch MaxToolErrorIter() on each call to .solve(). But if you don't want to, you can set this to False and in case there is a MaxToolErrorIter then the .solve() method will return None and won't throw. This might be cleaner to catch if you don't want to try/catch every call to .solve().  But be wary, this has not been tested extensively, yet, and is a behavior that might change in the near future.
    llm_stops_by_itself : bool
        Only useful when the task is part of a GroupSolve(). This signal the assigned LLM that it will have to stop talking by its onw means and is not only bound to a simple max iteration stop.
    use_self_reflection : bool
        Only useful when the task is part of a GroupSolve(). Allows to keep the self reflection process done by the LLM in the next GS iteration. May be useful if the LLM has problems with reasoning.
    forget : bool
        When this task has finished resolving and this is set to False, the Agent won't remember it happened. Useful when doing yes/no questions for routing purposes and no need to keep the answer in the history.
    uuid : str
        An identifier uniq to this task. Mostly used internally, but you can use it to keep track of things if you wish.

    Methods
    ----------
    add_tool(self, tool: Tool) -> None:
    solve(self) -> Message | None:
    """

    def __init__(self, prompt: str, agent: Agent, json_output=False, tools: List[Tool] = None,
                 raise_when_max_tool_error_iter: bool = True, llm_stops_by_itself: bool = False,
                 use_self_reflection=False, forget=False) -> None:
        """
        Returns a Task instance.
        @param prompt: str: The task to solve. It is the prompt given to the assigned LLM
        @param agent: str: The agent assigned to this task
        @param json_output: bool: If True, will force the LLM to answer as JSON. Its using Ollama json mode for now. We shall see how to implement that on other inference backends. Either way you should ask for a JSON output in the task prompt.
        @param tools:  List[Tool]: A list of tools that the LLM will get access to when trying to solve this task. This means that tools are not bound to the LLM itself but to the task. This provides more flexibility and less confusion to the LLM as it gets access to the tools it needs in relation to the task at hand.
        @param raise_when_max_tool_error_iter: bool: You should try/catch MaxToolErrorIter() on each call to .solve(). But if you don't want to, you can set this to False and in case there is a MaxToolErrorIter then the .solve() method will return None and won't throw. This might be cleaner to catch if you don't want to try/catch every call to .solve().  But be wary, this has not been tested extensively, yet, and is a behavior that might change in the near future.
        @param llm_stops_by_itself: bool: Only useful when the task is part of a GroupSolve(). This signal the assigned LLM that it will have to stop talking by its onw means and is not only bound to a simple max iteration stop.
        @param use_self_reflection: bool: Only useful when the task is part of a GroupSolve(). Allows to keep the self reflection process done by the LLM in the next GS iteration. May be useful if the LLM has problems with reasoning.
        @param forget: bool: When this task has finished resolving and this is set to True, the Agent won't remember it happened. Useful when doing yes/no questions for routing purposes and no need to keep the answer in the history.
        """
        self.task: str = prompt
        self.agent: Agent = agent
        self.json_output: bool = json_output
        self.tools: List[Tool] = tools if tools is not None else []
        self.raise_when_max_tool_error_iter: bool = raise_when_max_tool_error_iter  # Not a huge fan. Maybe add a callbak that would be called when we reach max iter instead of raising
        self.llm_stops_by_itself: bool = llm_stops_by_itself
        self.use_self_reflection: bool = use_self_reflection
        self.forget: bool = forget
        self._uuid: str = str(uuid.uuid4())

        # Only used when @forget is True
        self.save_history: History | None = None

    @property
    def uuid(self) -> str:
        """
        Read only property that references the current task with a unique id
        @return: str : A unique task id
        """
        return self._uuid

    def add_tool(self, tool: Tool) -> None:
        """
        Adds a Tool() to the list of tools to be used in this task. Note that is mostly syntactic sugar as you can append a tool to the member variable yourself.
        @param tool: Tool : A tool that will be given to the LLM when it tries to solve the task
        @return: None
        """
        self.tools.append(tool)

    def solve(self) -> Message | None:
        """
        This will call the assigned LLM to perform the inference on the prompt define in the task. The LLM will try to solve the task. If it has tools it will use them and multiple calls will
        probably be made to the inference server. Yacana uses percussive maintenance  to force the LLM to obey.
        This method may raise 'MaxToolErrorIter()' if either the tool is not used correctly or if Yacana fails at calling it correctly.
        @return: Message : The last message from the LLM
        """
        if self.forget is True:
            self.save_history: History = copy.deepcopy(self.agent.history)
        try:
            answer: Message = self.agent._interact(self.task, json_output=self.json_output, tools=self.tools)
        except MaxToolErrorIter as e:
            if self.raise_when_max_tool_error_iter:
                self._reset_agent_history()
                raise MaxToolErrorIter(e.message)
            else:
                self._reset_agent_history()
                # By default, we would raise but if @raise_when_max_tool_error_iter is False we simply return None.
                # An uncaught exceptions would be fatal so returning None is a good way to continue execution after a failure.
                return None  #@todo What happens in GroupSolve is the no raise option is used ?
        self._reset_agent_history()
        return answer

    def _reset_agent_history(self):
        if self.forget is True:
            self.agent.history = self.save_history
