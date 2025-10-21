import json
import logging
from json import JSONDecodeError
from typing import List, Type, T, Dict, Callable
from pydantic import BaseModel

from .exceptions import MaxToolErrorIter, ToolError, UnknownResponseFromLLM
from .history import OpenAiToolCallingMessage, OpenAIFunctionCallingMessage, OllamaToolCallingMessage
from .tool import Tool
from .base_tool_caller import BaseToolCaller
from .utils import AgentType


class OpenAiToolCaller(BaseToolCaller):

    def propose_tool(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None, tags: List[str] | None = None):
        self.propose_tools(task, tools, json_output, structured_output, medias, streaming_callback, task_runtime_config, tags)

    def propose_tools(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None, tags: List[str] | None = None):
        self.agent._chat(self.agent.history, task, medias=medias, json_output=json_output, structured_output=structured_output, tools=tools)
        if isinstance(self.agent.history.get_last_message(), OpenAIFunctionCallingMessage):
            for tool_call in self.agent.history.get_last_message().tool_calls:
                tool = next((tool for tool in tools if tool.tool_name == tool_call.name), None)
                if tool is None:
                    raise ValueError(f"Tool {tool_call.name} not found in tools list")
                logging.debug("Found tool: %s", tool.tool_name)
                func_arguments: dict = {}
                try:
                    if isinstance(tool_call.arguments, dict):
                        func_arguments = tool_call.arguments
                    else:
                        func_arguments: dict = json.loads(tool_call.arguments)
                        if not isinstance(func_arguments, dict): # In case of multi JSON encoding layers, arguments mays still not be a dict
                            raise ValueError(f"Still not a valid dict after doing json.loads on arguments.")
                except Exception as e:
                    raise UnknownResponseFromLLM(f"Tool {tool_call.name} arguments are not a valid dict object: `{tool_call.arguments}`. Type: {type(tool_call.arguments)}. Original error: {e}.")
                tool_output: str = self._call_openai_tool(tool, func_arguments)
                if self.agent._agent_type == AgentType.OPENAI:
                    self.agent.history.add_message(OpenAiToolCallingMessage(tool_output, tool_call.call_id, tags=self.agent._tags))
                elif self.agent._agent_type == AgentType.OLLAMA:
                    self.agent.history.add_message(OllamaToolCallingMessage(tool_output, tool_call.name, tags=self.agent._tags))

            logging.info(f"[PROMPT][To: {self.agent.name}]: Retrying with original task and tools answer: '{task}'")
            self.agent._chat(self.agent.history, None, medias=medias, json_output=json_output, structured_output=structured_output, streaming_callback=streaming_callback)

    def _call_openai_tool(self, tool: Tool, function_args: Dict) -> str:
        """
        Executes a tool call and handles any errors that occur.

        Parameters
        ----------
        tool : Tool
            The tool to execute.
        function_args : Dict
            The arguments to pass to the tool function.

        Returns
        -------
        str
            The output from the tool execution.

        Raises
        ------
        MaxToolErrorIter
            If too many errors occur during tool execution.
        """
        max_call_error: int = tool.max_call_error
        max_custom_error: int = tool.max_custom_error
        tool_output: str = ""

        while True:
            try:
                if tool.is_mcp:
                    tool_output: str = tool.function_ref(tool_name=tool.tool_name, arguments=function_args)
                else:
                    tool_output: str = tool.function_ref(**function_args)
                if tool_output is None:
                    tool_output = f"Tool returned 'None' so LLM won't be asked to reflect on the tool result."
                else:
                    tool_output = str(tool_output)
                logging.info(f"[TOOL_RESPONSE][{tool.tool_name}]: {tool_output}\n")
                break
            except (ToolError, TypeError, JSONDecodeError) as e:
                if type(e) is ToolError or type(e) is JSONDecodeError:
                    logging.warning(f"Tool '{tool.tool_name}' raised an error\n")
                    max_custom_error -= 1
                    tool_output = e.message
                elif type(e) is TypeError:
                    logging.warning(f"Yacana failed to call tool '{tool.tool_name}' correctly based on the LLM output\n")
                    tool_output = str(e)
                    max_call_error -= 1

                if max_custom_error < 0:
                    raise MaxToolErrorIter(
                        f"Too many errors were raise by the tool '{tool.tool_name}'. Stopping after {tool.max_custom_error} errors. You can change the maximum errors a tool can raise in the Tool constructor with @max_custom_error.")
                if max_call_error < 0:
                    raise MaxToolErrorIter(
                        f"Too many errors occurred while trying to call the python function by Yacana (tool name: {tool.tool_name}). Stopping after {tool.max_call_error} errors. You can change the maximum call error in the Tool constructor with @max_call_error.")
                self.agent._chat(self.agent.history, f"The tool returned an error: `{tool_output}`\nUsing this error message, fix the JSON you generated.")
        return tool_output
