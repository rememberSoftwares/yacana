import inspect
import json
import logging
from enum import Enum
from typing import List, Callable, Dict, Any, Tuple

from .exceptions import IllogicalConfiguration, McpBadToolConfig
from .function_to_json_schema import function_to_json_with_pydantic
from .history import History, MessageRole, OllamaUserMessage


class ToolType(Enum):
    """
    <b>ENUM:</b> ToolType

    How a tool must be presented to the LLM so it can be called.

    Attributes
    ----------
    OPENAI : str
        Tool calling will follow the OpenAi function calling format.
    YACANA : str
        Tool calling will follow the Yacana function calling format available to all LLMs.
    """
    OPENAI = "OPENAI"
    YACANA = "YACANA"


class Tool:
    """
    A class representing a tool that can be used by an LLM to perform specific tasks.

    A tool is a classic python function that can be assigned to a task. The LLM in charge
    of solving the task will have access to the tool and may try to use it or not
    depending on the configuration provided.

    Parameters
    ----------
    tool_name : str
        A name for the tool. Should be concise and related to what the tool does.
    function_description : str
        A description for the tool. Should be concise and related to what the tool does.
        May contain an example of how to use. Refer to the documentation.
    function_ref : Callable
        The reference to a python function that will be called with parameters provided by the LLM.
    optional : bool, optional
        Allows to a certain extent the LLM to choose to use the given tool or not depending on the task to solve.
        Defaults to False.
    usage_examples : List[dict], optional
        A list of python dictionary examples of how the tool should be called.
        The examples will be given to the LLM to help it call the tool correctly.
        Use if the LLM struggles to call the tool successfully. Defaults to an empty list.
    max_custom_error : int, optional
        The max errors a tool can raise.
        A tool should raise a ToolError(...) exception with a detailed explanation of why it failed.
        The LLM will get the exception message and try again, taking into account the new knowledge it gained from the error.
        When reaching the max iteration the MaxToolErrorIter() exception is thrown and the task is stopped. Defaults to 5.
    max_call_error : int, optional
        The max number of times Yacana can fail to call a tool correctly.
        Note that Yacana uses the parameters given to the LLM to call the tool so if they are invalid then Yacana will have a hard time to fix the situation.
        You should try to give examples to the LLM on how to call the tool either in the tool description or when using the @usage_examples attribute to help the model.
        Defaults to 5.
    shush : bool, optional
        If True, the tool won't warn anymore about the unsupported tool setting optional=True and using Ollama + OpenAi style tool calling.
        Defaults to False.

    Attributes
    ----------
    tool_name : str
        The name of the tool.
    function_description : str
        A description of the tool's functionality.
    function_ref : Callable
        Function reference that the tool will call.
    optional : bool
        Indicates if the tool is optional.
    usage_examples : List[dict]
        A list of usage examples for the tool. The dict keys should match the function parameters.
    max_custom_error : int
        Maximum number of custom errors (raised from the function) allowed before stopping the task.
    max_call_error : int
        Maximum number of call errors (eg: python can't find the function) allowed before stopping the task.
    is_mcp: bool
        Is this tool a local tool or an MCP tool.
    shush : bool
        If True, suppresses warnings about unsupported optional tool configurations with Ollama.
    tool_type: ToolType
        The tool execution style. Either use the Yacana style (default) or follow the OpenAi style.

    Raises
    ------
    IllogicalConfiguration
        If max_custom_error or max_call_error is less than 0.
    """

    def __init__(self, tool_name: str, function_description: str, function_ref: Callable, optional: bool = False,
                 usage_examples: List[dict] | None = None, max_custom_error: int = 5, max_call_error: int = 5,
                 tool_type: ToolType = ToolType.YACANA, mcp_input_schema: dict = None, shush=False) -> None:
        self.tool_name: str = tool_name
        self.function_description: str = function_description
        self.function_ref: Callable = function_ref
        self.is_mcp: bool = mcp_input_schema is not None
        self.optional: bool = optional
        self.usage_examples: List[dict] = usage_examples if usage_examples is not None else []
        self.mcp_input_schema: dict = mcp_input_schema

        if mcp_input_schema is not None:
            self._openai_function_schema: dict = self._function_to_json_with_mcp(mcp_input_schema)
            params: List[Tuple[str, str]] = self.input_shema_to_prototype(mcp_input_schema)
            self._function_prototype: str = self.tool_name + "(" + ", ".join([f"{name}: {type_}" for name, type_ in params]) + ")"
            self._function_args: List[str] = [param[0] for param in params]
        else:
            self._openai_function_schema: dict = self._function_to_json_with_pydantic()
            self._function_prototype: str = Tool._extract_prototype(function_ref)
            self._function_args: List[str] = Tool._extract_parameters(function_ref)

        self.max_custom_error: int = max_custom_error
        self.max_call_error: int = max_call_error
        self.tool_type: ToolType = tool_type
        self.shush = shush

        if max_custom_error < 0 or max_call_error < 0:
            raise IllogicalConfiguration("@max_custom_error and @max_call_error must be > 0")
        if " " in self.tool_name:
            logging.warning(f"Tool name {self.tool_name} contains spaces. Some inference servers may not support it. We recommend you use CamelCase instead.")

    def input_shema_to_prototype(self, input_shema: dict) -> List[Tuple[str, str]]:
        """
        Converts the input schema to a function prototype string.

        Parameters
        ----------
        input_shema : dict
            The input schema to convert.

        Returns
        -------
         List[(str, str)]
            tuple[0] is the param name and tuple[1] is the param type.
        """
        if input_shema.get("type") != "object" or "properties" not in input_shema:
            raise McpBadToolConfig(f"For tool '{self.tool_name}' from source MCP : Input schema must be an object with properties.")

        result: List[Tuple[str, str]] = []
        for param_name, param_info in input_shema["properties"].items():
            param_type = param_info.get("type", "Any")
            result.append((param_name, param_type))

        return result

    def _get_examples_as_history(self, tags: List[str]) -> History:
        """
        Convert the tool's usage examples into a conversation history format.
        This is multi shot prompting based on examples given to the tool.

        Parameters
        ----------
        tags : List[str]
            A list of tags to add to the messages.

        Returns
        -------
        History
            A history object containing the tool usage examples formatted as a conversation.
        """
        history = History()

        for example in self.usage_examples:
            tmp = ", ".join([f"{key} is {value}" for key, value in example.items()])
            history.add_message(OllamaUserMessage(MessageRole.USER,
                                f"For training purpose let's try calling the tool {self.tool_name} with theses parameter{'s' if len(example.items()) > 1 else ''}: {tmp}", tags=tags))
            history.add_message(OllamaUserMessage(MessageRole.ASSISTANT, json.dumps(example), tags=tags))
        if len(self.usage_examples) > 0:
            history.add_message(OllamaUserMessage(MessageRole.USER,
                                f"{'These were all' if len(self.usage_examples) > 1 else 'This was a'} great tool call{'s' if len(self.usage_examples) > 1 else ''}", tags=tags))
            history.add_message(OllamaUserMessage(MessageRole.ASSISTANT, "Great ! I understand how it works.", tags=tags))
        return history

    @staticmethod
    def _extract_prototype(func: Callable) -> str:
        """
        Extract the function prototype as a string.

        Parameters
        ----------
        func : Callable
            The function to extract the prototype from.

        Returns
        -------
        str
            The function prototype as a string, including the function name and signature.
        """
        # Get the function's signature
        sig = inspect.signature(func)
        # Format the signature as a string and returns it
        return f"{func.__name__}{sig}"

    @staticmethod
    def _extract_parameters(func: Callable) -> List[str]:
        """
        Extract the parameter names from a function's signature.

        Parameters
        ----------
        func : Callable
            The function to extract parameters from.

        Returns
        -------
        List[str]
            A list of parameter names from the function's signature.
        """
        signature = inspect.signature(func)
        # Access the parameters
        parameters = signature.parameters
        # Extract the parameter names into a list
        return [param_name for param_name in parameters]

    def _function_to_json_with_pydantic(self) -> Dict:
        """
        Convert the function to a JSON schema using Pydantic.

        This method generates a JSON schema for the function that can be used by
        OpenAI's function calling API. The schema is stored in the
        _openai_function_schema attribute.
        """
        return function_to_json_with_pydantic(self.tool_name, self.function_description, self.function_ref)

    def _function_to_json_with_mcp(self, input_shema: dict) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.function_description,
                "parameters": input_shema,
                "strict": True
            }
        }

    @staticmethod
    def bulk_tool_type_update(tools: List['Tool'], tool_type: ToolType) -> None:
        """
        Update the tool type for a list of tools.
        !Warning!: the tool type will remain the same until it is changed otherwise.
        This is different from mcp.get_tools_as(<tool_type>) which returns a copy of the tools.

        Parameters
        ----------
        tools : List[Tool]
            The list of tools to update.
        tool_type : ToolType
            The new tool type to set for all tools in the list.
        """
        for tool in tools:
            tool.tool_type = tool_type
