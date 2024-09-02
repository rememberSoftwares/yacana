import inspect
import json
from typing import List, Callable

from .exceptions import IllogicalConfiguration
from .history import History, Message, MessageRole


class Tool:
    """ A tool is a classic python function that can be assigned to a task. The LLM in charge of solving the task will
    have access to the tool and may try to use it or not depending on the configuration you provided.

    Attributes
    ----------
    tool_name : str
        A name for the tool. Should be concise and related to what the tool does.
    function_description : str
        A description for the tool. Should be concise and related to what the tool does. May contain an example of how to use. Refer to the documentation.
    function_ref : Callable
        The reference to a python function that will be called with parameters provided by the LLM.
    optional : bool
        Allows to a certain extent the LLM to choose to use the given tool or not depending on the task to solve.
    usage_examples : List[dict]
        A list of python dictionary examples of how the tool should be called. The examples will be given to the LLM to help it call the tool correctly. Use if the LLM struggles to call the tool successfully.
    max_custom_error : int
        The max errors a tool can raise. A tool should raise a ToolError(...) exception with a detailed explanation of why it failed. The LLM will get the exception message and try again, taking into account the new knowledge it gained from the error. When reaching the max iteration the MaxToolErrorIter() exception is thrown and the task is stopped.
    max_call_error
        The max number of times Yacana can fail to call a tool correctly. Note that Yacana uses the parameters given to the LLM to call the tool so if they are invalid then Yacana will have a hard time to fix the situation. You should try to give examples to the LLM on how to call the tool either in the tool description or when using the @usage_examples attribute to help the model.
    """

    def __init__(self, tool_name: str, function_description: str, function_ref: Callable, optional: bool = False,
                 usage_examples: List[dict] | None = None, max_custom_error: int = 5, max_call_error: int = 5) -> None:
        self.tool_name: str = tool_name
        self.function_description: str = function_description
        self.function_ref: Callable = function_ref
        self.optional: bool = optional
        self._function_prototype: str = Tool._extract_prototype(function_ref)
        self._function_args: List[str] = Tool._extract_parameters(function_ref)
        self.usage_examples: List[dict] = usage_examples if usage_examples is not None else []
        self.max_custom_error: int = max_custom_error
        self.max_call_error: int = max_call_error
        if max_custom_error < 0 or max_call_error < 0:
            raise IllogicalConfiguration("@max_custom_error and @max_call_error must be > 0")
        # Unused for now as it poses pb when there are multiple tools. We lack of a tool parent object that could store this information.
        #self.post_tool_prompt: str | None = post_tool_prompt_reflection

    def _get_examples_as_history(self):
        history = History()

        for example in self.usage_examples:
            tmp = ", ".join([f"{key} is {value}" for key, value in example.items()])
            history.add(Message(MessageRole.USER,
                                f"For training purpose let's try calling the tool {self.tool_name} with theses parameter{'s' if len(example.items()) > 1 else ''}: {tmp}"))
            history.add(Message(MessageRole.ASSISTANT, json.dumps(example)))
        if len(self.usage_examples) > 0:
            history.add(Message(MessageRole.USER,
                                f"{'These were all' if len(self.usage_examples) > 1 else 'This was a'} great tool call{'s' if len(self.usage_examples) > 1 else ''}"))
            history.add(Message(MessageRole.ASSISTANT, "Great ! I understand how it works."))
        return history

    @staticmethod
    def _extract_prototype(func: Callable):
        # Get the function's signature
        sig = inspect.signature(func)
        # Format the signature as a string and returns it
        return f"{func.__name__}{sig}"

    @staticmethod
    def _extract_parameters(func) -> List[str]:
        signature = inspect.signature(func)
        # Access the parameters
        parameters = signature.parameters
        # Extract the parameter names into a list
        return [param_name for param_name in parameters]
