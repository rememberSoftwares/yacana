import json
import logging
import uuid

from ollama import Client, ChatResponse
from typing import List, Type, Any, T, Dict, Callable, Mapping, Tuple
from collections.abc import Iterator
from pydantic import BaseModel

from .generic_agent import GenericAgent
from .model_settings import OllamaModelSettings
from .utils import Dotdict, AgentType
from .exceptions import IllogicalConfiguration, TaskCompletionRefusal
from .history import HistorySlot, GenericMessage, MessageRole, History, OllamaUserMessage, OllamaStructuredOutputMessage, OllamaTextMessage, ToolCallFromLLM, OpenAIFunctionCallingMessage
from .tool import Tool, ToolType
from .constants import PROMPT_TAG, RESPONSE_TAG

logger = logging.getLogger(__name__)


class OllamaAgent(GenericAgent):
    """
    Representation of an LLM agent that interacts with the Ollama inference server.

    This class provides ways to interact with the LLM, but it should not be controlled directly.
    Instead, it should be assigned to a Task(). When a task is required to be solved, the agent will
    interact with the prompt inside the task and output an answer. This class is more about
    configuring the agent than interacting with it.

    Parameters
    ----------
    name : str
        Name of the agent. Use something short and meaningful that doesn't contradict the system prompt.
    model_name : str
        Name of the LLM model that will be sent to the inference server (e.g., 'llama:3.1' or 'mistral:latest').
    system_prompt : str | None, optional
        Defines the way the LLM will behave (e.g., "You are a pirate" to have it talk like a pirate).
        Defaults to None.
    endpoint : str, optional
        The Ollama endpoint URL. Defaults to "http://127.0.0.1:11434".
    headers : dict, optional
        Custom headers to be sent with the inference request. Defaults to None.
    model_settings : OllamaModelSettings, optional
        All settings that Ollama currently supports as model configuration. Defaults to None.
    runtime_config : Dict | None, optional
        Runtime configuration for the agent. Defaults to None.
    thinking_tokens : Tuple[str, str] | None, optional
        A tuple containing the start and end tokens of a thinking LLM. For instance, "<think>" and "</think>" for Deepseek-R1.
        Setting this prevents the framework from getting sidetracked during the thinking steps and helps maintain focus on the final result.
    structured_thinking : bool, optional
        If True, Yacana will use structured_output internally to get better accuracy. If your LLM doesn't support structured_output set this to False.
        Defaults to True.
    **kwargs
        Additional keyword arguments passed to the parent class.

    Attributes
    ----------
    _agent_type : AgentType
        Type of the Agent to circumvent partial import when determining agent's type at runtime.

    Raises
    ------
    IllogicalConfiguration
        If model_settings is not an instance of OllamaModelSettings.
    """

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str = "http://127.0.0.1:11434", headers=None, model_settings: OllamaModelSettings = None, runtime_config: Dict | None = None, thinking_tokens: Tuple[str, str] | None = None, structured_thinking=True, **kwargs) -> None:
        model_settings = OllamaModelSettings() if model_settings is None else model_settings
        if not isinstance(model_settings, OllamaModelSettings):
            raise IllogicalConfiguration("model_settings must be an instance of OllamaModelSettings.")
        self._agent_type: AgentType = AgentType.OLLAMA
        super().__init__(name, model_name, model_settings, system_prompt=system_prompt, endpoint=endpoint, api_token="", headers=headers, runtime_config=runtime_config, history=kwargs.get("history", None), task_runtime_config=kwargs.get("task_runtime_config", None), thinking_tokens=thinking_tokens, structured_thinking=structured_thinking)

    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None, tags: List[str] | None = None) -> GenericMessage:
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
        medias : List[str] | None
            Optional list of media files.
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
        IllogicalConfiguration
            If streaming is requested with tool usage.
        """
        self._set_correct_tool_caller(tools)
        self._tags = tags if tags is not None else []
        tools: List[Tool] = [] if tools is None else tools
        self.task_runtime_config = task_runtime_config if task_runtime_config is not None else {}

        if len(tools) == 0:
            self._chat(self.history, task, medias=medias, json_output=json_output, structured_output=structured_output, streaming_callback=streaming_callback)
        elif len(tools) == 1:
            self.tool_caller.propose_tool(task, tools, json_output, structured_output, medias, streaming_callback, task_runtime_config, tags)
        elif len(tools) > 1:
            self.tool_caller.propose_tools(task, tools, json_output, structured_output, medias, streaming_callback, task_runtime_config, tags)

        return self.history.get_last_message()

    """def _ollama_tool_names_conversion(self, tools: List[Tool]) -> List[Tool]:
        for tool in tools:
            if not tool.is_mcp and tool.tool_type == ToolType.OPENAI:
                if tool.function_ref.__name__ != tool.tool_name:
                    logging.warning(f"Ollama expects the tool name to be the same as the function name. Tool '{tool.tool_name}' will be renamed automatically to '{tool.function_ref.__name__}'.")
                    tool.update_tool_name_to_match_function_name()

        return tools"""

    def _stream(self) -> None:
        """
        Placeholder for streaming functionality.

        This method is currently not implemented.
        """
        pass

    @staticmethod
    def _get_expected_output_format(json_output: bool, structured_output: Type[BaseModel] | None) -> dict[str, Any] | str:
        """
        Determines the expected output format (JSON, structured_output, etc.) based on the configuration set by the Task.

        Parameters
        ----------
        json_output : bool
            Whether to output JSON.
        structured_output : Type[BaseModel] | None
            Optional structured output type.

        Returns
        -------
        dict[str, Any] | str
            The expected output format.
        """
        if structured_output:
            return structured_output.model_json_schema()
        elif json_output:
            return 'json'
        else:
            return ''

    def _ollama_tool_call_to_json(self, tool_calls: List) -> List:
        tools_as_json: List = []
        if tool_calls:
            # There may be multiple tool calls in the response
            for tool in tool_calls:
                tools_as_json.append({"name": tool.function.name, "arguments": tool.function.arguments})
        return tools_as_json

    def _response_to_json(self, response: Any) -> str:
        """
        Converts an Ollama response object to JSON format. Used for setting raw_llm_json in HistorySlot.

        Parameters
        ----------
        response : Any
            The response object to convert.

        Returns
        -------
        str
            The JSON string representation of the response.

        Raises
        ------
        TypeError
            If the conversion to JSON fails.
        """
        try:
            result: Dict[str, Any] = {
                'model': getattr(response, 'model', None),
                'created_at': getattr(response, 'created_at', None),
                'done': getattr(response, 'done', None),
                'done_reason': getattr(response, 'done_reason', None),
                'total_duration': getattr(response, 'total_duration', None),
                'load_duration': getattr(response, 'load_duration', None),
                'prompt_eval_count': getattr(response, 'prompt_eval_count', None),
                'prompt_eval_duration': getattr(response, 'prompt_eval_duration', None),
                'eval_count': getattr(response, 'eval_count', None),
                'eval_duration': getattr(response, 'eval_duration', None),
            }

            # Extract 'message' if present
            message = getattr(response, 'message', None)
            if message is not None:
                result['message'] = {
                    'role': getattr(message, 'role', None),
                    'content': getattr(message, 'content', None),
                    'images': getattr(message, 'images', None),
                    'tool_calls': self._ollama_tool_call_to_json(getattr(message, 'tool_calls', None))
                }

            # Return the JSON string representation
            return json.dumps(result, indent=4)
        except Exception as e:
            raise TypeError(f"Failed to convert response to JSON: {e}")

    def _dispatch_chunk_if_streaming(self, chat_response: ChatResponse | Iterator[ChatResponse], streaming_callback: Callable | None) -> Dict | ChatResponse | Iterator[ChatResponse]:
        """
        Handles streaming responses by dispatching chunks to the callback.

        Parameters
        ----------
        chat_response : ChatResponse | Iterator[ChatResponse]
            The completion response or iterator.
        streaming_callback : Callable | None
            Optional callback for streaming responses.

        Returns
        -------
        Dict | ChatResponse | Iterator[ChatResponse]
            The processed response.

        Raises
        ------
        TaskCompletionRefusal
            If the streaming response contains no data.
        """
        # If we are not streaming, we return the Ollama message directly.
        if streaming_callback is None:
            return chat_response
        all_chunks = ""
        for chunk in chat_response:
            if chunk['message']['content'] is not None:
                all_chunks += chunk['message']['content']
                streaming_callback(chunk['message']['content'])
            else:
                raise TaskCompletionRefusal("Streaming LLMs response returned no data (content == None).")
        return Dotdict({
                    "message": {
                        "content": all_chunks,
                    }
                }
            )

    def _is_tool_calling(self, message) -> bool:
        """
        Checks if the message contains tool calls.

        Parameters
        ----------
        message : Any
            The message to check.

        Returns
        -------
        bool
            True if the choice contains tool calls, False otherwise.
        """
        return message.tool_calls and len(message.tool_calls) > 0

    def _chat(self, history: History, task: str | None, medias: List[str] | None = None, json_output: bool = False, structured_output: Type[T] | None = None, save_to_history: bool = True, tools: List[Tool] | None = None, streaming_callback: Callable | None = None) -> GenericMessage:
        """
        Main chat method that handles communication with the Ollama server.

        Parameters
        ----------
        history : History
            The conversation history.
        task : str | None
            The task to execute.
        medias : List[str] | None, optional
            Optional list of media files. Defaults to None.
        json_output : bool, optional
            Whether to output JSON. Defaults to False.
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
            The response content.
        """

        if task:
            logging.info(f"[PROMPT][To: {self.name}]: {task}")
            question_slot = history.add_message(OllamaUserMessage(MessageRole.USER, task, tags=self._tags + [PROMPT_TAG], medias=medias, structured_output=structured_output))

        if tools is not None and len(tools) > 0:
            for tool in tools:
                if tool.tool_type == ToolType.OPENAI and tool.optional is False and tool.shush is False:
                    logging.warning(f"You chose to use the OpenAI style tool calling with the OllamaAgent for the tool '{tool.tool_name}'. This tool is set by default as optional=False (hence making it mandatory to use). Note that Ollama does NOT support setting tools optional status on tools! They are all optional by default and this cannot be changed. Yacana may in the future mitigate this issue. If this is important for you please open an issue on the Yacana Github. You can hide this warning by setting `shush=True` in the Tool constructor.")
        client = Client(host=self.endpoint, headers=self.headers)
        params = {
            "model": self.model_name,
            "messages": history.get_messages_as_dict(),
            "format": self._get_expected_output_format(json_output, structured_output),
            "stream": True if streaming_callback is not None else False,
            "options": self.model_settings.get_settings(),
            **({"tools": [tool._openai_function_schema for tool in tools]} if tools is not None else {}),
            **self.runtime_config,
            **self.task_runtime_config
        }
        logging.debug("Runtime parameters before inference: %s", str(params))
        chat_response: ChatResponse | Iterator[ChatResponse] = client.chat(**params)
        response = self._dispatch_chunk_if_streaming(chat_response, streaming_callback)
        logging.debug("Inference output: %s", str(response))
        if structured_output is not None:
            logging.debug("Response assessment is structured output")
            answer_slot: HistorySlot = history.add_message(OllamaStructuredOutputMessage(MessageRole.ASSISTANT, str(response['message']['content']), structured_output.model_validate_json(response['message']['content']), tags=self._tags + [RESPONSE_TAG]))
        elif self._is_tool_calling(response['message']):
            logging.debug("Response assessment is tool calling")
            tool_calls: List[ToolCallFromLLM] = []
            for tool_call in response['message']["tool_calls"]:
                tool_calls.append(ToolCallFromLLM(str(uuid.uuid4()), tool_call.function.name, tool_call.function.arguments))
                logging.debug("Tool info : Name= %s, Arguments= %s", tool_call.function.name, tool_call.function.arguments)
            answer_slot: HistorySlot = history.add_message(OpenAIFunctionCallingMessage(tool_calls, tags=self._tags))
        else:
            answer_slot: HistorySlot = history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, response['message']['content'], tags=self._tags + [RESPONSE_TAG]))

        self.task_runtime_config = {}
        answer_slot.set_raw_llm_json(self._response_to_json(response))

        logging.info(f"[AI_RESPONSE][From: {self.name}]: {answer_slot.get_message().get_as_pretty()}")

        last_message = history.get_last_message()
        if save_to_history is False:
            if task:
                history.delete_slot(question_slot)
            history.delete_slot(answer_slot)
        return last_message
