import copy
from json import JSONDecodeError
import json
import logging
from typing import List, Type, T, Dict, Callable
from pydantic import BaseModel

from .base_tool_caller import BaseToolCaller
from .structured_outputs import UseTool, MakeAnotherToolCall
from .exceptions import MaxToolErrorIter, ToolError, IllogicalConfiguration
from .history import GenericMessage, MessageRole, History, OllamaUserMessage, OllamaTextMessage, Message, HistorySlot
from .tool import Tool
from .constants import PROMPT_TAG


class YacanaToolCaller(BaseToolCaller):

    def propose_tool(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None, tags: List[str] | None = None):
        local_history: History = copy.deepcopy(self.agent.history)
        tool: Tool = tools[0]

        tool_definition = str(tool._function_prototype + " - " + tool.function_description)
        tool_ack_prompt = f"I give you the following tool definition that you {'must' if tool.optional is False else 'may'} use to fulfill a future task: {tool_definition}. Please acknowledge the given tool."
        self.agent._chat(local_history, tool_ack_prompt)

        # This section checks whether we need a tool or not. If not we call the LLM like if tools == 0 and exit the function.
        if tool.optional is True:
            task_outputting_prompt = f'You have a task to solve. In your opinion, is using the tool "{tool.tool_name}" relevant to solve the task or not ? The task is:\n{task}'
            self.agent._chat(local_history, task_outputting_prompt, medias=medias)

            if self.agent.structured_thinking is True:
                tool_use_router_prompt: str = "To summarize in one word your previous answer. Do you wish to use the tool or not ? Respond by true or false using valid JSON."
                tool_use_ai_confirmation: GenericMessage = self.agent._chat(local_history, tool_use_router_prompt, save_to_history=False, structured_output=UseTool)
                if tool_use_ai_confirmation.structured_output.useTool is False:
                    return self.agent._chat(self.agent.history, task, medias=medias, json_output=json_output, structured_output=structured_output)
            else:
                tool_use_router_prompt: str = "To summarize in one word your previous answer. Do you wish to use the tool or not ? Respond ONLY by 'yes' or 'no'."
                tool_use_ai_confirmation: GenericMessage = self.agent._chat(local_history, tool_use_router_prompt, save_to_history=False)
                if "yes" in self.agent._strip_thinking_tags(tool_use_ai_confirmation.content.lower()):
                    return self.agent._chat(self.agent.history, task, medias=medias, json_output=json_output, structured_output=structured_output)

        # If getting here the tool call is inevitable
        local_history._concat_history(tool._get_examples_as_history(self.agent._tags))
        tool_use: str = 'To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect the argument type of each parameter. In our case, the tool call you must use must look like that: ' + json.dumps(
            {key: ("arg " + str(index)) for index, key in enumerate(tool._function_args)}) + f"\nNow that I showed you examples on how the tool is used, you have a task to solve. The task is:\n<task>{task}</task>\nPlease output the tool call as valid JSON."

        if len(tool._function_args) <= 0:
            logging.info(f"[PROMPT][To: {self.agent.name}]: {tool_use}")
            local_history.add_message(Message(MessageRole.USER, tool_use, tags=self.agent._tags))
            answer_slot: HistorySlot = local_history.add_message(Message(MessageRole.ASSISTANT, "{}", tags=self.agent._tags))  # Empty function calling because no function arguments are needed.
            logging.info(f"[AI_RESPONSE][From: {self.agent.name}]: {answer_slot.get_message().get_as_pretty()}")
        else:
            self.agent._chat(local_history, tool_use, medias=medias, json_output=True)  # !!Actual function calling
        tool_output: str | None = self._tool_call(local_history, tool)  # !!Actual tool calling
        logging.debug(f"Tool output: {tool_output}\n")
        if tool_output is not None:  # If the tool outputted something then we can ask the LLM to reflect on it.
            local_history.add_message(OllamaTextMessage(MessageRole.USER, f"The tool '{tool.tool_name}' gave the following output:\n{tool_output}", tags=self.agent._tags))
            local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, f"Thank you. I successfully called the tool and got the result. What should I do with it now ?", tags=self.agent._tags))
            self.agent._chat(local_history, f"Based on the previous tool output. Solve the initial task. The task was: {task}.", streaming_callback=streaming_callback)
            # Bellow is the final reconciliation of the main history composed of the initial task and the reflection from the LLM on the tool output.
            self.agent.history.add_message(OllamaTextMessage(MessageRole.USER, task, tags=self.agent._tags + [PROMPT_TAG]))
            self.agent.history.add_message((OllamaTextMessage(MessageRole.ASSISTANT, local_history.get_last_message().content, tags=self.agent._tags + [PROMPT_TAG])))
        else:  # The tool didn't output anything so the LLM won't be asked to reflect on anything more.
            self.agent.history.add_message(OllamaTextMessage(MessageRole.USER, task, tags=self.agent._tags + [PROMPT_TAG]))
            self.agent.history.add_message((OllamaTextMessage(MessageRole.ASSISTANT, "The task was solved successfully.", tags=self.agent._tags + [PROMPT_TAG])))

    def propose_tools(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None, tags: List[str] | None = None):
        at_least_one_tool_has_outputted: bool = False
        local_history = copy.deepcopy(self.agent.history)

        tools_presentation: str = "* " + "\n* ".join([
            f"Name: '{tool.tool_name}' - Usage: {tool._function_prototype} - Description: {tool.function_description}"
            for tool in tools])
        tool_ack_prompt = f"You have access to this list of tools definitions you can use to fulfill tasks :\n{tools_presentation}\nPlease acknowledge the given tools."
        self.agent._chat(local_history, tool_ack_prompt)

        tool_use_decision: str = f"You have a task to solve. I will give it to you between these tags `<task></task>`. However, your actual job is to decide if you need to use any of the available tools to solve the task or not. If you do need tools then output their names. The task to solve is <task>{task}</task> So, would any tools be useful in relation to the given task ?"
        self.agent._chat(local_history, tool_use_decision, medias=medias)

        if self.agent.structured_thinking is True:
            tool_router: str = "In order to summarize your previous answer: Did you chose to use any tools ? Answer as valid JSON."
            ai_may_use_tools: GenericMessage = self.agent._chat(local_history, tool_router, save_to_history=False, structured_output=UseTool)
            use_tool = ai_may_use_tools.structured_output.useTool is True
        else:
            tool_router: str = "In order to summarize your previous answer: Did you chose to use any tools ? Respond ONLY by 'yes' or 'no'."
            ai_may_use_tools: GenericMessage = self.agent._chat(local_history, tool_router, save_to_history=False)
            use_tool = "yes" in self.agent._strip_thinking_tags(ai_may_use_tools.content.lower())

        if use_tool:
            self.agent.history.add_message(OllamaUserMessage(MessageRole.USER, task, tags=self.agent._tags))
            self.agent.history.add_message(
                OllamaTextMessage(MessageRole.ASSISTANT, "I should use tools related to the task to solve it correctly.", tags=self.agent._tags))
            while True:
                tool: Tool = self._choose_tool_by_name(local_history, tools)
                tool_training_history = copy.deepcopy(local_history)

                tool_training_history._concat_history(tool._get_examples_as_history(self.agent._tags))

                tool_use: str = 'To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect the argument type of each parameter. In our case, the tool call you must use must look like that: ' + json.dumps(
                    {key: ("arg " + str(i)) for i, key in enumerate(tool._function_args)}) + "\nNow that I showed you examples on how the tool is used it's your turn. Output the tool as valid JSON."
                if len(tool._function_args) <= 0:
                    logging.info(f"[PROMPT][To: {self.agent.name}]: {tool_use}")
                    tool_training_history.add_message(Message(MessageRole.USER, tool_use, tags=self.agent._tags))
                    answer_slot: HistorySlot = tool_training_history.add_message(Message(MessageRole.ASSISTANT, "{}", tags=self.agent._tags))  # Empty function calling because no function arguments are needed.
                    logging.info(f"[AI_RESPONSE][From: {self.agent.name}]: {answer_slot.get_message().get_as_pretty()}")
                else:
                    self.agent._chat(tool_training_history, tool_use, medias=medias, json_output=True)  # !!Actual function calling
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
            self.agent._chat(self.agent.history, task, medias=medias, json_output=json_output)
        if at_least_one_tool_has_outputted is True:
            self.agent._chat(self.agent.history, f"Based on the previous tools output. Solve the initial task. The task was: {task}.", streaming_callback=streaming_callback)

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
            ai_tool_choice: str = self.agent._chat(local_history, tool_choose).content
            ai_tool_choice = self.agent._strip_thinking_tags(ai_tool_choice).strip(" \n").lower()

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
                self.agent.model_settings.reset()
                return found_tools[0]
            elif len(found_tools) >= 2:
                logging.warning("More than one tool was proposed. Trying again.\n")

            # No tool or too many tools found
            local_history.add_message(OllamaUserMessage(MessageRole.USER,
                                                        "You outputted more than one tool name. Let's try again with only outputting the single tool name to use and no other.", tags=self.agent._tags))
            logging.info(f"[prompt]: You outputted more than one tool name. Let's try again with only outputting the single tool name to use and no other.\n")
            local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT,
                                                        "I'm sorry. I know I must ONLY output the name of the one tool I wish to use. Let's try again and I'll output only the single tool name this time!", tags=self.agent._tags))
            logging.info(f"[AI_RESPONSE]: I'm sorry. I know I must ONLY output the name of the one tool I wish to use. Let's try again and I'll output only the single tool name this time!\n")
            max_tool_name_use_iter += 1
            # Forcing LLM to be less chatty and more focused
            if max_tool_name_use_iter >= 2:
                if self.agent.model_settings.temperature is None:
                    self.agent.model_settings.temperature = 2
                self.agent.model_settings.temperature = self.agent.model_settings.temperature / 2
            if max_tool_name_use_iter >= 3:
                self.agent.model_settings.tfs_z = 2
            if max_tool_name_use_iter >= 4:
                # Getting the longer tool name and setting max token output to this value x 1.5. This should reduce output length of AI's response.
                self.agent.model_settings.num_predict = len([max([tool.tool_name for tool in tools], key=len)]) * 1.5

        self.agent.model_settings.reset()
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
                function_args: dict = json.loads(self.agent._strip_thinking_tags(tool_training_history.get_last_message().content))
                if tool.is_mcp:
                    tool_output: str = tool.function_ref(tool_name=tool.tool_name, arguments=function_args)
                else:
                    tool_output: str = tool.function_ref(**function_args)
                if tool_output is None:
                    logging.info(f"[TOOL_RESPONSE][{tool.tool_name}]: Tool returned 'None' so LLM won't be asked to reflect on the tool result.")
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
                self.agent._chat(tool_training_history, fix_your_shit_prompt, json_output=True)
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
        self.agent.history.add_message(OllamaUserMessage(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON.", tags=self.agent._tags))
        local_history.add_message(OllamaUserMessage(MessageRole.USER, f"Output the tool '{tool.tool_name}' as valid JSON.", tags=self.agent._tags))

        # Master history + local history get fake ASSISTANT prompt calling the tool correctly
        self.agent.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, tool_training_history.get_last_message().content, tags=self.agent._tags))
        local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, tool_training_history.get_last_message().content, tags=self.agent._tags))

        # Master history + local history get fake USER prompt with the answer of the tool
        self.agent.history.add_message(OllamaTextMessage(MessageRole.USER, f"The tool '{tool.tool_name}' gave the following output:\n{tool_output}", tags=self.agent._tags))
        local_history.add_message(OllamaTextMessage(MessageRole.USER, f"The tool '{tool.tool_name}' gave the following output:\n{tool_output}", tags=self.agent._tags))

        # Master history + local history get fake ASSISTANT prompt that acknowledge the tool output
        self.agent.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, f"Now that the tool '{tool.tool_name}' gave its answer. What should I do next ?", tags=self.agent._tags))
        local_history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, f"Now that the tool '{tool.tool_name}' gave its answer. What should I do next ?", tags=self.agent._tags))


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
        self.agent.history.add_message(OllamaUserMessage(MessageRole.USER, initial_task, tags=self.agent._tags + [PROMPT_TAG]))
        self.agent.history.add_message((OllamaUserMessage(MessageRole.ASSISTANT, final_response, tags=self.agent._tags + [PROMPT_TAG])))


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
        ai_tool_continue_answer: str = self.agent._chat(local_history, tool_continue_prompt).content

        # Syncing with global history
        self.agent.history.add_message(OllamaUserMessage(MessageRole.USER, tool_continue_prompt, tags=self.agent._tags))
        self.agent.history.add_message(OllamaTextMessage(MessageRole.ASSISTANT, ai_tool_continue_answer, tags=self.agent._tags))

        if self.agent.structured_thinking is True:
            tool_confirmation_prompt = "To summarize your previous answer: Do you need to make another tool call ? Output your answer as valid JSON."
            ai_tool_continue_confirmation: GenericMessage = self.agent._chat(local_history, tool_confirmation_prompt, save_to_history=False, structured_output=MakeAnotherToolCall)
            tool_continuation = ai_tool_continue_confirmation.structured_output.makeAnotherToolCall is True
        else:
            tool_confirmation_prompt = "To summarize your previous answer: Do you need to make another tool call ? Respond ONLY by 'yes' or 'no'."
            ai_tool_continue_confirmation: GenericMessage = self.agent._chat(local_history, tool_confirmation_prompt, save_to_history=False)
            tool_continuation = "yes" in self.agent._strip_thinking_tags(ai_tool_continue_confirmation.content.lower())

        if tool_continuation:
            logging.info("Continuing tool calls loop")
            return True
        else:
            logging.info("LLM didn't request another tool. Exiting tool calls loop")
            return False
