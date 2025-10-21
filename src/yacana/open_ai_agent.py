import json
import logging
from json import JSONDecodeError
from openai import OpenAI, Stream
from typing import List, Mapping, Type, Any, Literal, T, Dict, Callable
from collections.abc import Iterator
from openai.types.chat.chat_completion import Choice, ChatCompletion
from pydantic import BaseModel

from openai.types.chat import ChatCompletionChunk

from .generic_agent import GenericAgent
from .model_settings import OpenAiModelSettings
from .utils import Dotdict
from .exceptions import MaxToolErrorIter, ToolError, IllogicalConfiguration, TaskCompletionRefusal, UnknownResponseFromLLM
from .history import OpenAIToolCallingMessage, HistorySlot, GenericMessage, MessageRole, ToolCallFromLLM, OpenAIFunctionCallingMessage, OpenAITextMessage, History, OpenAIStructuredOutputMessage, OpenAIUserMessage
from .tool import Tool
from .constants import PROMPT_TAG, RESPONSE_TAG

logger = logging.getLogger(__name__)


class OpenAiAgent(GenericAgent):
    """
    Representation of an LLM agent that interacts with the OpenAI API.

    This class provides ways to interact with the LLM, but it should not be controlled directly.
    Instead, it should be assigned to a Task(). When a task is required to be solved, the agent will
    interact with the prompt inside the task and output an answer. This class is more about
    configuring the agent than interacting with it.

    Parameters
    ----------
    name : str
        Name of the agent. Use something short and meaningful that doesn't contradict the system prompt.
    model_name : str
        Name of the LLM model that will be sent to the inference server (e.g., 'gpt-4' or 'gpt-3.5-turbo').
    system_prompt : str | None, optional
        Defines the way the LLM will behave (e.g., "You are a pirate" to have it talk like a pirate).
        Defaults to None.
    endpoint : str | None, optional
        The OpenAI endpoint URL. Defaults to None (uses OpenAI's default endpoint).
    api_token : str, optional
        The API token for authentication. Defaults to an empty string.
    headers : dict, optional
        Custom headers to be sent with the inference request. Defaults to None.
    model_settings : OpenAiModelSettings, optional
        All settings that OpenAI currently supports as model configuration. Defaults to None.
    runtime_config : Dict | None, optional
        Runtime configuration for the agent. Defaults to None.
    thinking_tokens : Tuple[str, str] | None, optional
        A tuple containing the start and end tokens of a thinking LLM. For instance, "<think>" and "</think>" for Deepseek-R1.
        Setting this prevents the framework from getting sidetracked during the thinking steps and helps maintain focus on the final result.

    Raises
    ------
    IllogicalConfiguration
        If model_settings is not an instance of OpenAiModelSettings.
    """

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str | None = None,
                 api_token: str = "cant_be_empty", headers=None, model_settings: OpenAiModelSettings = None, runtime_config: Dict | None = None, thinking_tokens: tuple[str, str] | None = None, hugging_face_repo_name: str | None = None, hugging_face_token: str | None = None, **kwargs) -> None:
        if api_token == "":
            logging.warning(f"Empty api_token provided. This will most likely clash with the underlying inference library. You should probably set this to any non empty string.")
        model_settings = OpenAiModelSettings() if model_settings is None else model_settings
        if not isinstance(model_settings, OpenAiModelSettings):
            raise IllogicalConfiguration("model_settings must be an instance of OpenAiModelSettings.")
        super().__init__(name, model_name, model_settings, system_prompt=system_prompt, endpoint=endpoint, api_token=api_token, headers=headers, runtime_config=runtime_config, history=kwargs.get("history", None), task_runtime_config=kwargs.get("task_runtime_config", None), thinking_tokens=thinking_tokens, hugging_face_repo_name=hugging_face_repo_name, hugging_face_token=hugging_face_token)
        if self.api_token == "":
            logging.warning("OpenAI requires an API token to be set.")

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
                tool_output: str = tool.function_ref(**function_args)
                if tool_output is None:
                    tool_output = f"Tool {tool.tool_name} was called successfully. It didn't return anything."
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
                self._chat(self.history, f"The tool returned an error: `{tool_output}`\nUsing this error message, fix the JSON you generated.")
        return tool_output

    def _update_tool_definition(self, tools: List[Tool]) -> None:
        """
        Updates the OpenAI function schema for each tool.

        Parameters
        ----------
        tools : List[Tool]
            List of tools to update.
        """
        tools: List[Tool] = [] if tools is None else tools
        for tool in tools:
            if tool._openai_function_schema is None:
                tool._function_to_json_with_pydantic()

    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, images: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None, tags: List[str] | None = None) -> GenericMessage:
        """
        Main interaction method that handles task execution with optional tool usage.

        Parameters
        ----------
        task : str
            The task to execute.
        tools : List[Tool]
            List of available tools.
        json_output : bool
            Whether to output JSON.
        structured_output : Type[BaseModel] | None
            Optional structured output type.
        images : List[str] | None
            Optional list of image files.
        streaming_callback : Callable | None, optional
            Optional callback for streaming responses. Defaults to None.
        task_runtime_config : Dict | None, optional
            Optional runtime configuration for the task. Defaults to None.
        tags : List[str] | None, optional
            Optional list of tags. Defaults to None.
        Returns
        -------
        GenericMessage
            The response message from the agent.

        Raises
        ------
        ValueError
            If a requested tool is not found in the tools list.
        """
        self._tags = tags if tags is not None else []
        self._update_tool_definition(tools)
        self.task_runtime_config = task_runtime_config if task_runtime_config is not None else {}

        if len(tools) == 0:
            self._chat(self.history, task, medias=images, json_output=json_output, structured_output=structured_output, streaming_callback=streaming_callback)
        elif len(tools) > 0:
            self._chat(self.history, task, medias=images, json_output=json_output, structured_output=structured_output, tools=tools)
            if isinstance(self.history.get_last_message(), OpenAIFunctionCallingMessage):
                for tool_call in self.history.get_last_message().tool_calls:
                    tool = next((tool for tool in tools if tool.tool_name == tool_call.name), None)
                    if tool is None:
                        raise ValueError(f"Tool {tool_call.name} not found in tools list")
                    logging.debug("Found tool: %s", tool.tool_name)
                    tool_output: str = self._call_openai_tool(tool, tool_call.arguments)
                    self.history.add_message(OpenAIToolCallingMessage(tool_output, tool_call.call_id, tags=self._tags))
                logging.info(f"[PROMPT][To: {self.name}]: Retrying with original task and tools answer: '{task}'")
                self._chat(self.history, None, medias=images, json_output=json_output, structured_output=structured_output, streaming_callback=streaming_callback)
        return self.history.get_last_message()

    def _is_structured_output(self, choice: Choice) -> bool:
        """
        Checks if the choice contains structured output.

        Parameters
        ----------
        choice : Choice
            The choice to check.

        Returns
        -------
        bool
            True if the choice contains structured output, False otherwise.
        """
        return hasattr(choice.message, "parsed") and choice.message.parsed is not None

    def _is_tool_calling(self, choice: Choice) -> bool:
        """
        Checks if the choice contains tool calls.

        Parameters
        ----------
        choice : Choice
            The choice to check.

        Returns
        -------
        bool
            True if the choice contains tool calls, False otherwise.
        """
        return hasattr(choice.message, "tool_calls") and choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0

    def _is_common_chat(self, choice: Choice) -> bool:
        """
        Checks if the choice contains a common chat message.

        Parameters
        ----------
        choice : Choice
            The choice to check.

        Returns
        -------
        bool
            True if the choice contains a common chat message, False otherwise.
        """
        return hasattr(choice.message, "content") and choice.message is not None

    def _dispatch_chunk_if_streaming(self, completion: ChatCompletion | Stream[ChatCompletionChunk], streaming_callback: Callable | None) -> Dict | Mapping[str, Any] | Iterator[Mapping[str, Any]]:
        """
        Handles streaming responses by dispatching chunks to the callback.

        Parameters
        ----------
        completion : ChatCompletion | Stream[ChatCompletionChunk]
            The completion response or stream.
        streaming_callback : Callable | None
            Optional callback for streaming responses.

        Returns
        -------
        Dict | Mapping[str, Any] | Iterator[Mapping[str, Any]]
            The processed response.

        Raises
        ------
        TaskCompletionRefusal
            If the streaming response contains a refusal.
        """
        if streaming_callback is None:
            return completion
        all_chunks = ""
        for chunk in completion:
            if chunk.choices[0].delta.refusal in (False, None):
                if chunk.choices[0].delta.content is not None:
                    all_chunks += chunk.choices[0].delta.content
                    streaming_callback(chunk.choices[0].delta.content)
            else:
                raise TaskCompletionRefusal("Got a refusal from the LLM. This is not supported in streaming mode.")
        return Dotdict({
            "choices": [
                {
                    "message": {
                        "content": all_chunks,
                    }
                }
            ]
        })

    def _chat(self, history: History, task: str | None, medias: List[str] | None = None, json_output=False, structured_output: Type[T] | None = None, save_to_history: bool = True, tools: List[Tool] | None = None,
                  streaming_callback: Callable | None = None) -> GenericMessage:
        """
        Main chat method that handles communication with the OpenAI API.

        Parameters
        ----------
        history : History
            The conversation history.
        task : str | None
            The task to execute.
        medias : List[str] | None, optional
            Optional list of media files. Defaults to None.
        json_output : bool, optional
            Whether to output JSON (best effort). Defaults to False.
        structured_output : Type[T] | None, optional
            Optional structured output type. Defaults to None.
        save_to_history : bool, optional
            Whether to save the response to history. Defaults to True.
        tools : List[Tool] | None, optional
            Optional list of tools. Defaults to None.
        streaming_callback : Callable | None, optional
            Optional callback for streaming responses. Defaults to None.

        Returns
        -------
        GenericMessage
            The response message

        Raises
        ------
        ValueError
            If an unknown response type is received from the OpenAI API.
        TaskCompletionRefusal
            If the model refuses to complete the task.
        """
        if task:
            logging.info(f"[PROMPT][To: {self.name}]: {task}")
            question_slot = history.add_message(OpenAIUserMessage(MessageRole.USER, task, tags=self._tags + [PROMPT_TAG], medias=medias, structured_output=structured_output))
        # Extracting all json schema from tools, so it can be passed to the OpenAI API
        all_function_calling_json = [tool._openai_function_schema for tool in tools] if tools else []

        tool_choice_option = self._find_right_tool_choice_option(tools)
        response_format = self._find_right_output_format(structured_output, json_output)

        client = OpenAI(
            api_key=self.api_token,
            base_url=self.endpoint
        )

        params = {
            "model": self.model_name,
            "messages": history.get_messages_as_dict(),
            **({"stream": True} if streaming_callback is not None else {}),
            **({"response_format": response_format} if response_format is not None else {}),
            **({"tools": all_function_calling_json} if len(all_function_calling_json) > 0 else {}),
            **({"tool_choice": tool_choice_option} if len(all_function_calling_json) > 0 else {}),
            **self.model_settings.get_settings(),
            **self.runtime_config,
            **self.task_runtime_config
        }
        logging.debug("Runtime parameters before inference: %s", str(params))

        answer_slot = HistorySlot()
        if structured_output is not None:
            response = client.beta.chat.completions.parse(**params)
        else:
            response = client.chat.completions.create(**params)
            response = self._dispatch_chunk_if_streaming(response, streaming_callback)

        self.task_runtime_config = {}
        answer_slot.set_raw_llm_json(response.model_dump_json())
        logging.debug("Inference output: %s", response.model_dump_json(indent=2))

        for choice in response.choices:

            if self._is_structured_output(choice):
                logging.debug("Response assessment is structured output")
                if choice.message.refusal is not None:
                    raise TaskCompletionRefusal(choice.message.refusal)  # Refusal key is only available for structured output but also doesn't work very well
                answer_slot.add_message(OpenAIStructuredOutputMessage(MessageRole.ASSISTANT, choice.message.content, choice.message.parsed, tags=self._tags + [RESPONSE_TAG]))

            elif self._is_tool_calling(choice):
                logging.debug("Response assessment is tool calling")
                tool_calls: List[ToolCallFromLLM] = []
                for tool_call in choice.message.tool_calls:
                    tool_calls.append(ToolCallFromLLM(tool_call.id, tool_call.function.name, json.loads(tool_call.function.arguments)))
                    logging.debug("Tool info : Id= %s, Name= %s, Arguments= %s", tool_call.id, tool_call.function.name, tool_call.function.arguments)
                answer_slot.add_message(OpenAIFunctionCallingMessage(tool_calls, tags=self._tags))

            elif self._is_common_chat(choice):
                logging.debug("Response assessment is classic chat answer")
                answer_slot.add_message(OpenAITextMessage(MessageRole.ASSISTANT, choice.message.content, tags=self._tags + [RESPONSE_TAG]))
            else:
                raise UnknownResponseFromLLM("Unknown response from OpenAI API")

        logging.info(f"[AI_RESPONSE][From: {self.name}]: {answer_slot.get_message().get_as_pretty()}")
        last_message = answer_slot.get_message()
        if save_to_history is False:
            if task:
                history.delete_slot(question_slot)
            history.delete_slot(answer_slot)
        else:
            history.add_slot(answer_slot)
        return last_message


    def _find_right_tool_choice_option(self, tools: List[Tool] | None) -> Literal["none", "auto", "required"]:
        """
        Determines the appropriate tool choice option based on tool configurations.

        Parameters
        ----------
        tools : List[Tool] | None
            List of tools to analyze.

        Returns
        -------
        Literal["none", "auto", "required"]
            The appropriate tool choice option:
            - "none" if no tools are provided
            - "auto" if all tools are optional
            - "required" if all tools are required

        Raises
        ------
        IllogicalConfiguration
            If there is a mix of required and optional tools.
        """
        if tools is None:
            return "none"

        all_optional = all(tool.optional for tool in tools)
        all_required = all(not tool.optional for tool in tools)

        if all_optional:
            return "auto"
        elif all_required:
            return "required"
        else:
            raise IllogicalConfiguration("OpenAI does not allow mixing required and optional tools.")

    def _find_right_output_format(self, structured_output: Type[T] | None, json_output: bool) -> Any:
        """
        Determines the appropriate output format based on the configuration.
        It determines if we want to get a structured output or a best effort JSON object.

        Parameters
        ----------
        structured_output : Type[T] | None
            Optional structured output type.
        json_output : bool
            Whether to output JSON.

        Returns
        -------
        Any
            The appropriate output format:
            - structured_output if provided
            - {"type": "json_object"} if json_output is True
            - None otherwise
        """
        if structured_output is not None:
            return structured_output
        elif json_output is True:
            return {"type": "json_object"}  # This is NOT the "structured output" feature, but only "best effort" to get a JSON object (as string)
        else:
            return None
