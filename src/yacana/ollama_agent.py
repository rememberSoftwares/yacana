import copy
from json import JSONDecodeError
import json
import logging
from ollama import Client
from typing import List, Type, Any, T, Dict, Callable, Mapping, Tuple
from collections.abc import Iterator
from pydantic import BaseModel

from .generic_agent import GenericAgent
from .model_settings import OllamaModelSettings
from .structured_outputs import UseTool, MakeAnotherToolCall
from .utils import Dotdict
from .exceptions import MaxToolErrorIter, ToolError, IllogicalConfiguration, TaskCompletionRefusal
from .history import HistorySlot, GenericMessage, MessageRole, History, OllamaUserMessage, OllamaStructuredOutputMessage, OllamaTextMessage
from .tool import Tool
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

    **kwargs
        Additional keyword arguments passed to the parent class.

    Raises
    ------
    IllogicalConfiguration
        If model_settings is not an instance of OllamaModelSettings.
    """

    def __init__(self, name: str, model_name: str, system_prompt: str | None = None, endpoint: str = "http://127.0.0.1:11434", headers=None, model_settings: OllamaModelSettings = None, runtime_config: Dict | None = None, thinking_tokens: Tuple[str, str] | None = None, hugging_face_repo_name: str | None = None, hugging_face_token: str | None = None, **kwargs) -> None:
        model_settings = OllamaModelSettings() if model_settings is None else model_settings
        if not isinstance(model_settings, OllamaModelSettings):
            raise IllogicalConfiguration("model_settings must be an instance of OllamaModelSettings.")
        super().__init__(name, model_name, model_settings, system_prompt=system_prompt, endpoint=endpoint, api_token="", headers=headers, runtime_config=runtime_config, history=kwargs.get("history", None), task_runtime_config=kwargs.get("task_runtime_config", None), thinking_tokens=thinking_tokens, hugging_face_repo_name=hugging_face_repo_name, hugging_face_token=hugging_face_token)

    def _choose_tool_by_name(self, local_history: History, tools: List[Tool]) -> Tool:
        """
        Selects a tool from the available tools based on the LLM's choice.

        Parameters
        ----------
        local_history : History
            The conversation history to use for tool selection.
        tools : List[Tool]
            List of available tools to choose from.

        Returns
        -------
        Tool
            The selected tool.

        Raises
        ------
        MaxToolErrorIter
            If the LLM fails to choose a tool after multiple attempts.
        IllogicalConfiguration
            If there are duplicate tool names.
        """
        # Checking if all tool names are uniq
        tool_names = [tool.tool_name.lower() for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise IllogicalConfiguration("More than one tool have the same name. Tool names must be unique.")

        max_tool_name_use_iter: int = 0
        while max_tool_name_use_iter < 5:

            tool_choose: str = f"You can only use one tool at a time. From this list of tools which one do you want to use: [{', '.join([tool.tool_name for tool in tools])}] ? You must answer ONLY with the single tool name. Nothing else."
            ai_tool_choice: str = self._chat(local_history, tool_choose).content
            ai_tool_choice = self._strip_thinking_tags(ai_tool_choice).strip(" \n").lower()

            print("ai tool choice = ", ai_tool_choice)

            found_tools: List[Tool] = []

            for tool in tools:
                # If tool name is present somewhere in AI response
                if tool.tool_name.lower() in ai_tool_choice:
                    # If tool name is not an exact match in AI response
                    if ai_tool_choice != tool.tool_name.lower():
                        logging.warning("Tool choice was not an exact match but a substring match\n")
                    found_tools.append(tool)

            # If there was more than 1 tool name in the AI answer we cannot be sure what tool it chose. So we try again.
            if len(found_tools) == 1:
                self.model_settings.reset()
                return found_tools[0]
            elif len(found_tools) >= 2:
                logging.warning("More than one tool was proposed. Trying again.\n")

            # No tool or too many tools found
            local_history.add_message(OllamaUserMessage(MessageRole.USER,
                                                        "You outputted more than one tool name. Let's try again with only outputting the single tool name to use and no other.", tags=self._tags))
            logging.info(f"[prompt]: You outputted more than one tool name. Let's try again with only outputting the single tool name to use and no other.\n")
            local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT,
                                                        "I'm sorry. I know I must ONLY output the name of the one tool I wish to use. Let's try again and I'll output only the single tool name this time!", tags=self._tags))
            logging.info(f"[AI_RESPONSE]: I'm sorry. I know I must ONLY output the name of the one tool I wish to use. Let's try again and I'll output only the single tool name this time!\n")
            max_tool_name_use_iter += 1
            # Forcing LLM to be less chatty and more focused
            if max_tool_name_use_iter >= 2:
                if self.model_settings.temperature is None:
                    self.model_settings.temperature = 2
                self.model_settings.temperature = self.model_settings.temperature / 2
            if max_tool_name_use_iter >= 3:
                self.model_settings.tfs_z = 2
            if max_tool_name_use_iter >= 4:
                # Getting the longer tool name and setting max token output to this value x 1.5. This should reduce output length of AI's response.
                self.model_settings.num_predict = len([max([tool.tool_name for tool in tools], key=len)]) * 1.5

        self.model_settings.reset()
        raise MaxToolErrorIter("[ERROR] LLM did not choose a tool from the list despite multiple attempts.")

    def _tool_call(self, tool_training_history: History, tool: Tool) -> str | None:
        """
        Executes a tool call and handles any errors that occur.

        Parameters
        ----------
        tool_training_history : History
            The conversation history containing the tool call parameters.
        tool : Tool
            The tool to execute.

        Returns
        -------
        str | None
            The output from the tool execution. If None then no post-processing on the tool output by the LLM will be done.

        Raises
        ------
        MaxToolErrorIter
            If too many errors occur during tool execution.
        """
        max_call_error: int = tool.max_call_error
        max_custom_error: int = tool.max_custom_error
        tool_output: str = ""

        while True:
            additional_prompt_help: str = ""
            try:
                args: dict = json.loads(self._strip_thinking_tags(tool_training_history.get_last_message().content))
                tool_output: str = tool.function_ref(**args)
                if tool_output is None:
                    logging.info(f"[TOOL_RESPONSE][{tool.tool_name}]: None -> LLM won't be asked to reflect on the tool result.")
                    #tool_output = f"Tool {tool.tool_name} was called successfully. It didn't return anything."
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
                    additional_prompt_help = 'Remember that you must output ONLY the tool arguments as valid JSON. For instance: ' + str(
                        {key: ("arg " + str(i)) for i, key in enumerate(tool._function_args)})
                    max_call_error -= 1

                if max_custom_error < 0:
                    raise MaxToolErrorIter(
                        f"Too many errors were raised by the tool '{tool.tool_name}'. Stopping after {tool.max_custom_error} errors. You can change the maximum errors a tool can raise in the Tool constructor with @max_custom_error.")
                if max_call_error < 0:
                    raise MaxToolErrorIter(
                        f"Too many errors occurred while trying to call the python function by Yacana (tool name: {tool.tool_name}). Stopping after {tool.max_call_error} errors. You can change the maximum call error in the Tool constructor with @max_call_error.")

                fix_your_shit_prompt = f"The tool returned an error: `{tool_output}`\nUsing this error message, fix the JSON arguments you gave.\n{additional_prompt_help}"
                self._chat(tool_training_history, fix_your_shit_prompt, json_output=True)
        return tool_output

    def _reconcile_history_multi_tools(self, tool_training_history: History, local_history: History, tool: Tool, tool_output: str) -> None:
        """
        Reconciles the history for multiple tool calls.

        Parameters
        ----------
        tool_training_history : History
            The history containing the tool training.
        local_history : History
            The local conversation history.
        tool : Tool
            The tool that was called.
        tool_output : str
            The output from the tool execution.
        """
        # Master history + local history get fake USER prompt to ask for tool output
        self.history.add_message(OllamaUserMessage(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON.", tags=self._tags))
        local_history.add_message(OllamaUserMessage(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON.", tags=self._tags))

        # Master history + local history get fake ASSISTANT prompt calling the tool correctly
        self.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, tool_training_history.get_last_message().content, tags=self._tags))
        local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, tool_training_history.get_last_message().content, tags=self._tags))

        # Master history + local history get fake USER prompt with the answer of the tool
        self.history.add_message(OllamaTextMessage(MessageRole.USER, f"The tool '{tool.tool_name}' gave the following output:\n{tool_output}", tags=self._tags))
        local_history.add_message(OllamaTextMessage(MessageRole.USER, f"The tool '{tool.tool_name}' gave the following output:\n{tool_output}", tags=self._tags))

        # Master history + local history get fake ASSISTANT prompt that acknowledge the tool output
        self.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, f"Now that the tool '{tool.tool_name}' gave its answer. What should I do next ?", tags=self._tags))
        local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, f"Now that the tool '{tool.tool_name}' gave its answer. What should I do next ?", tags=self._tags))

    def _reconcile_history_solo_tool(self, initial_task: str, final_response: str) -> None:
        """
        Reconciles the history for a single tool call.

        Parameters
        ----------
        initial_task : str
            the task to solve from Task()
        final_response : str
            The output from the LLM to the initial task after the tool call.
        """
        self.history.add_message(OllamaUserMessage(MessageRole.USER, initial_task, tags=self._tags + [PROMPT_TAG]))
        #self.history.add_message(
        #    OllamaTextMessage(MessageRole.ASSISTANT,
        #                      f"I can use the tool '{tool.tool_name}' related to the task to solve it correctly.", tags=self._tags))
        #self.history.add_message(OllamaUserMessage(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON.", tags=self._tags))
        #self.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, last_tool_call, tags=self._tags))
        #self.history.add_message(OllamaTextMessage(MessageRole.USER, tool_output, tags=self._tags + [RESPONSE_TAG]))
        self.history.add_message((OllamaUserMessage(MessageRole.ASSISTANT, final_response, tags=self._tags + [PROMPT_TAG])))

    def _use_other_tool(self, local_history: History) -> bool:
        """
        Determines if another tool call is needed.

        Parameters
        ----------
        local_history : History
            The conversation history.

        Returns
        -------
        bool
            True if another tool call is needed, False otherwise.
        """
        tool_continue_prompt = "Now that the tool responded do you need to make another tool call ? Explain why and what the remaining steps are if any. DO NOT make the tool call now. Just think."
        ai_tool_continue_answer: str = self._chat(local_history, tool_continue_prompt).content

        # Syncing with global history
        self.history.add_message(OllamaUserMessage(MessageRole.USER, tool_continue_prompt, tags=self._tags))
        self.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, ai_tool_continue_answer, tags=self._tags))

        tool_confirmation_prompt = "To summarize your previous answer: Do you need to make another tool call ? Output your answer as valid JSON."
        ai_tool_continue_confirmation: GenericMessage = self._chat(local_history, tool_confirmation_prompt,
                                                  save_to_history=False, structured_output=MakeAnotherToolCall)

        if ai_tool_continue_confirmation.structured_output.makeAnotherToolCall is True:  # "yes" in self._strip_thinking_tags(ai_tool_continue_answer.lower()):
            logging.info("Continuing tool calls loop")
            return True
        else:
            logging.info("LLM didn't request another tool. Exiting tool calls loop")
            return False

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
        self._tags = tags if tags is not None else []
        tools: List[Tool] = [] if tools is None else tools
        self.task_runtime_config = task_runtime_config if task_runtime_config is not None else {}

        if len(tools) == 0:
            self._chat(self.history, task, medias=medias, json_output=json_output, structured_output=structured_output, streaming_callback=streaming_callback)

        elif len(tools) == 1:
            local_history = copy.deepcopy(self.history)
            tool: Tool = tools[0]

            tool_definition = str(tool._function_prototype + " - " + tool.function_description)
            tool_ack_prompt = f"I give you the following tool definition that you {'must' if tool.optional is False else 'may'} use to fulfill a future task: {tool_definition}. Please acknowledge the given tool."
            self._chat(local_history, tool_ack_prompt)

            # This section checks whether we need a tool or not. If not we call the LLM like if tools == 0 and exit the function.
            if tool.optional is True:
                task_outputting_prompt = f'You have a task to solve. In your opinion, is using the tool "{tool.tool_name}" relevant to solve the task or not ? The task is:\n{task}'
                self._chat(local_history, task_outputting_prompt, medias=medias)

                tool_use_router_prompt: str = "To summarize in one word your previous answer. Do you wish to use the tool or not ? Respond ONLY by 'yes' or 'no'."
                tool_use_ai_confirmation: GenericMessage = self._chat(local_history, tool_use_router_prompt, save_to_history=False, structured_output=UseTool)
                if tool_use_ai_confirmation.structured_output.useTool is False:
                #if not ("yes" in self._strip_thinking_tags(tool_use_ai_answer.lower())):  # Better than checking for "no" as a substring could randomly match
                    self._chat(self.history, task, medias=medias, json_output=json_output, structured_output=structured_output)  # Classic LLM call without tool usage
                    return self.history.get_last_message()

            # If getting here the tool call is inevitable
            local_history._concat_history(tool._get_examples_as_history(self._tags))
            tool_use: str = 'To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect the argument type of each parameter. In our case, the tool call you must use must look like that: ' + str(
                {key: ("arg " + str(i)) for i, key in enumerate(tool._function_args)}) + f"\nNow that I showed you examples on how the tool is used, you have a task to solve. The task is:\n<task>{task}</task>\nPlease output the tool call as valid JSON."
            self._chat(local_history, tool_use, medias=medias, json_output=True)  # !!Actual function calling
            tool_output: str | None = self._tool_call(local_history, tool)  # !!Actual tool calling
            logging.debug(f"Tool output: {tool_output}\n")
            if tool_output is not None:  # If the tool outputted something then we can ask the LLM to reflect on it.
                local_history.add_message(OllamaTextMessage(MessageRole.USER, f"The tool '{tool.tool_name}' gave the following output:\n{tool_output}", tags=self._tags))
                local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, f"Thank you. I successfully called the tool and got the result. What should I do with it now ?", tags=self._tags))
                self._chat(local_history, f"Based on the previous tool output. Solve the initial task. The task was: {task}.", streaming_callback=streaming_callback)
                # Bellow is the final reconciliation of the main history composed of the initial task and the reflection from the LLM on the tool output.
                self.history.add_message(OllamaTextMessage(MessageRole.USER, task, tags=self._tags + [PROMPT_TAG]))
                self.history.add_message((OllamaTextMessage(MessageRole.ASSISTANT, local_history.get_last_message().content, tags=self._tags + [PROMPT_TAG])))
            else:  # The tool didn't output anything so the LLM won't be asked to reflect on anything more.
                self.history.add_message(OllamaTextMessage(MessageRole.USER, task, tags=self._tags + [PROMPT_TAG]))
                self.history.add_message((OllamaTextMessage(MessageRole.ASSISTANT, "The task was solved successfully.", tags=self._tags + [PROMPT_TAG])))

        elif len(tools) > 1:
            at_least_one_tool_has_outputted: bool = False
            local_history = copy.deepcopy(self.history)

            tools_presentation: str = "* " + "\n* ".join([
                f"Name: '{tool.tool_name}' - Usage: {tool._function_prototype} - Description: {tool.function_description}"
                for tool in tools])
            tool_ack_prompt = f"You have access to this list of tools definitions you can use to fulfill tasks :\n{tools_presentation}\nPlease acknowledge the given tools."
            self._chat(local_history, tool_ack_prompt)

            tool_use_decision: str = f"You have a task to solve. I will give it to you between these tags `<task></task>`. However, your actual job is to decide if you need to use any of the available tools to solve the task or not. If you do need tools then output their names. The task to solve is <task>{task}</task> So, would any tools be useful in relation to the given task ?"
            self._chat(local_history, tool_use_decision, medias=medias)

            tool_router: str = "In order to summarize your previous answer: Did you chose to use any tools ? Answer as valid JSON."
            ai_may_use_tools: GenericMessage = self._chat(local_history, tool_router, save_to_history=False, structured_output=UseTool)

            if ai_may_use_tools.structured_output.useTool is True:  # "yes" in self._strip_thinking_tags(ai_may_use_tools.lower()):
                self.history.add_message(OllamaUserMessage(MessageRole.USER, task, tags=self._tags))
                self.history.add_message(
                    OllamaTextMessage(MessageRole.ASSISTANT, "I should use tools related to the task to solve it correctly.", tags=self._tags))
                while True:
                    tool: Tool = self._choose_tool_by_name(local_history, tools)
                    tool_training_history = copy.deepcopy(local_history)

                    tool_training_history._concat_history(tool._get_examples_as_history(self._tags))

                    tool_use: str = 'To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect the argument type of each parameter. In our case, the tool call you must use must look like that: ' + str(
                        {key: ("arg " + str(i)) for i, key in enumerate(tool._function_args)}) + "\nNow that I showed you examples on how the tool is used it's your turn. Output the tool as valid JSON."
                    self._chat(tool_training_history, tool_use, medias=medias, json_output=True)  # !!Actual function calling
                    tool_output: str | None = self._tool_call(tool_training_history, tool)  # !!Actual tool calling
                    if tool_output is not None:
                        at_least_one_tool_has_outputted = True
                    else:
                        tool_output = f"Tool was called successfully."
                    self._reconcile_history_multi_tools(tool_training_history, local_history, tool, tool_output)
                    use_other_tool: bool = self._use_other_tool(local_history)
                    if use_other_tool is True:
                        continue
                    else:
                        break
            else:
                # Getting here means that no tools were selected by the LLM and we act like tools == 0
                self._chat(self.history, task, medias=medias, json_output=json_output)
            if at_least_one_tool_has_outputted is True:
                self._chat(self.history, f"Based on the previous tools output. Solve the initial task. The task was: {task}.", streaming_callback=streaming_callback)
        return self.history.get_last_message()

    def _stream(self) -> None:
        """
        Placeholder for streaming functionality.

        This method is currently not implemented.
        """
        pass


    @staticmethod
    def _get_expected_output_format(json_output: bool, structured_output: Type[BaseModel] | None) -> dict[str, Any] | str:
        """
        Determines the expected output format based on the configuration.

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

    def _response_to_json(self, response: Any) -> str:
        """
        Converts a response object to JSON format.

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
                    'tool_calls': getattr(message, 'tool_calls', None)
                }

            # Return the JSON string representation
            return json.dumps(result, indent=4)
        except Exception as e:
            raise TypeError(f"Failed to convert response to JSON: {e}")

    def _dispatch_chunk_if_streaming(self, completion: Mapping[str, Any] | Iterator[Mapping[str, Any]], streaming_callback: Callable | None) -> Dict | Mapping[str, Any] | Iterator[Mapping[str, Any]]:
        """
        Handles streaming responses by dispatching chunks to the callback.

        Parameters
        ----------
        completion : Mapping[str, Any] | Iterator[Mapping[str, Any]]
            The completion response or iterator.
        streaming_callback : Callable | None
            Optional callback for streaming responses.

        Returns
        -------
        Dict | Mapping[str, Any] | Iterator[Mapping[str, Any]]
            The processed response.

        Raises
        ------
        TaskCompletionRefusal
            If the streaming response contains no data.
        """
        if streaming_callback is None:
            return completion
        all_chunks = ""
        for chunk in completion:
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

    def _chat(self, history: History, task: str | None, medias: List[str] | None = None, json_output = False, structured_output: Type[T] | None = None, save_to_history: bool = True, tools: List[Tool] | None = None, streaming_callback: Callable | None = None) -> GenericMessage:
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

        #print("what his getting = ", history.get_messages_as_dict())
        client = Client(host=self.endpoint, headers=self.headers)
        params = {
            "model": self.model_name,
            "messages": history.get_messages_as_dict(),
            "format": self._get_expected_output_format(json_output, structured_output),
            "stream": True if streaming_callback is not None else False,
            "options": self.model_settings.get_settings(),
            **self.runtime_config,
            **self.task_runtime_config
        }
        logging.debug("Runtime parameters before inference: %s", str(params))
        response = client.chat(**params)
        logging.debug("Inference output: %s", str(response))
        if structured_output is not None:
            answer_slot: HistorySlot = history.add_message(OllamaStructuredOutputMessage(MessageRole.ASSISTANT, str(response['message']['content']), structured_output.model_validate_json(response['message']['content']), tags=self._tags + [RESPONSE_TAG]))
        else:
            response = self._dispatch_chunk_if_streaming(response, streaming_callback)
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
