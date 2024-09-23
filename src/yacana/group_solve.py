import copy
from enum import Enum
from typing import List
import logging
from typing_extensions import Tuple

from .agent import Message, MessageRole
from .task import Task
from .exceptions import ReachedTaskCompletion, IllogicalConfiguration

logger = logging.getLogger(__name__)


class EndChatMode(Enum):
    """ All types of group chat completion.
    The difficulty in making agents talk to each other is not to have them talk but to have them stop talking.
    Note that only tasks that have the @llm_stops_by_itself=True are actually impacted by the mode set here.
    Use in conjunction with EndChat()

    Attributes
    ----------
    ALL_TASK_MUST_COMPLETE : str
        chat will continue going until all LLMs with @llm_stops_by_itself=True says they are finished (Set precise completion goals in the task prompt if you want this to actualy work).
    ONE_LAST_CHAT_AFTER_FIRST_COMPLETION : str
        One agent will have the opportunity to respond after the completion of one agent allowing it to answer one last time.
    ONE_LAST_GROUP_CHAT_AFTER_FIRST_COMPLETION : str
        All agents will have one last table turn to speak before exiting the chat after the first completion arrives
    END_CHAT_AFTER_FIRST_COMPLETION : str
        Immediately stops group chat after an agent has reached completion
    MAX_ITERATIONS_ONLY : str
        LLMs won't be asked if they have fulfilled their objectives but instead will loop until achieving max iteration. Max iteration can be set in the EndChat() class below.

    """
    ALL_TASK_MUST_COMPLETE = "ALL_TASK_MUST_COMPLETE"
    ONE_LAST_CHAT_AFTER_FIRST_COMPLETION = "ONE_LAST_CHAT_AFTER_FIRST_COMPLETION"
    ONE_LAST_GROUP_CHAT_AFTER_FIRST_COMPLETION = "ONE_LAST_GROUP_CHAT_AFTER_FIRST_COMPLETION"
    END_CHAT_AFTER_FIRST_COMPLETION = "END_CHAT_AFTER_FIRST_COMPLETION"
    MAX_ITERATIONS_ONLY = "MAX_ITERATIONS_ONLY"


class EndChat:
    """Defines the modality of how and when LLMs stop chatting.
    By default, when reaching the @max_iterations limit the chat will end.
    However, if a task has set @llm_stops_by_itself=True then you can use one of the EndChatMode values to specify how
    and when the chat stops after reaching the first completion.

    Attributes
    ----------
    mode : EndChatMode
        The modality to end a chat with multiple agents
    max_iterations : int
        The max number of iterations in a conversation. An iteration is complete when we get back to the first speaker
    """
    def __init__(self, mode: EndChatMode, max_iterations: int = 5):
        """
        Defines the modality of how and when LLMs stop chatting.
        :param mode: EndChatMode: The modality to end a chat with multiple agents
        :param max_iterations: int: The max number of iterations in a conversation. An iteration is complete when we get back to the first speaker
        """
        if not isinstance(mode, EndChatMode):
            raise ValueError(
                "@mode must be an instance of EndChatMode. For instance: EndChatMode.ALL_TASK_MUST_COMPLETE")
        if max_iterations <= 0:
            raise ValueError("@max_iterations must be > 0")
        self.mode: EndChatMode = mode
        self.max_iterations: int = max_iterations


class GroupSolve:
    """This class allows multiple agents to enter a conversation with each other. This is made different from other
    frameworks as it is not directly the agents that enter the group chat but the tasks. Note that each task has an LLM
    Agent bound to it though. However, this changes the approach you should have when creating the conversation prompts.
    If you are letting LLMs decide when to stop chatting then be sure to specify precise goals. For instance 'Your task
    is done when XXXX'. This way the LLM will reflect on that and make a decision if it has achieved its original Task.
    Note that dual chat is more efficient as each agent thinks it is talking to a human and this is how instruct models
    have been trained. When having a conversation with more than two agents (aka more than 2 tasks) then Yacana will
    force them into roleplay. This might degrade performance a bit but has the advantage of showing a clearer logs.
    Logging output for dual chat will be reworked shortly.

    Attributes
    ----------
    tasks : list[task]
        All tasks that must be solved during group chat
    mode : EndChatMode
        Defines the way to stop the conversation besides than topping max iteration
    max_iter : int
        Sets a max iteration counter to prevent infinite loops

    Methods
    ----------
    solve(self) -> None

    """

    def __init__(self, tasks: List[Task], end_chat: EndChat, reconcile_first_message: bool = False, shift_message_owner: Task = None, shift_message_content: str | None = None) -> None:
        """
        :param tasks: list[task] : All tasks that must be solved during group chat
        :param end_chat: EndChat : Defines the modality of how and when LLMs stop chatting.
        :param reconcile_first_message: bool : Should the first message from both LLMs be available to one another. Only useful in dual chat.
        :param shift_message_owner: Task : The Task to which the shift message should be assigned to. In the end it's rather the corresponding Agent than the Task that is involved here.
        :param shift_message_content: str : A custom message instead of using the opposite agent response as shift message content.
        """
        if tasks is None:
            raise ValueError("@tasks cannot be None. Must be a List of Task")
        if not isinstance(end_chat, EndChat):
            raise ValueError(
                "@end_chat_mode must be an instance of EndChat.")
        self.tasks: List[Task] = tasks
        self.mode: EndChatMode = end_chat.mode
        self.reconcile_first_message: bool = reconcile_first_message
        self.max_iter: int = end_chat.max_iterations
        self._once: bool = False
        self.shift_owner: Task = shift_message_owner
        self.shift_content: str | None = shift_message_content
        # Looks like this {"task_uid1": False, "task_uid2": False}
        self._tasks_completion_statuses = {cur_task.uuid: False for cur_task in self.tasks if
                                           cur_task.llm_stops_by_itself}

    def set_shift_message(self) -> tuple:
        """
        Assigning shift message to either Agents and sets a custom message if user gave one. If not, the opposite agent's ANSWER will be used as PROMPT (shift message).
        :return: (str, (Task, Task))
        """
        my_task1 = self.tasks[0]
        my_task2 = self.tasks[1]

        # Setting default: shift message will be assigned to task2
        if self.shift_owner is None:
            self.shift_owner = my_task2

        # Assigning shift message to either Agents and sets a custom message if user gave one. If not, the opposite agent's ANSWER will be used as PROMPT (shift message).
        if self.shift_owner is my_task1:
            if self.shift_content is None:
                last_generated_answer: str = my_task2.agent.history.get_last().content
            else:
                last_generated_answer: str = self.shift_content
            tasks_run_order: Tuple = (my_task1, my_task2)
        elif self.shift_owner is my_task2:
            if self.shift_content is None:
                last_generated_answer: str = my_task1.agent.history.get_last().content
            else:
                last_generated_answer: str = self.shift_content
            tasks_run_order: Tuple = (my_task2, my_task1)
        else:
            raise IllogicalConfiguration("@shift_message_owner parameter from GroupSolve() class must be an instance of Task present in the array of Task also given to the constructor.")
        return last_generated_answer, tasks_run_order

    def solve(self) -> None:
        """
        Starts the group chat and allows all LLMs to solve their assigned tasks. Note that 'dual chat' and '3 and more'
        chat have a different way of starting. Refer to the official documentation.
        @return: None
        """
        if self._once is True:
            raise IllogicalConfiguration("You cannot use the same GroupSolve twice. You must instantiate a new one.")
        self._once = True
        # When @exit_after_next_chat passes True we'll exit after the next chat with an agent
        exit_after_next_chat: bool = False
        # When @exit_after_next_chat changes from None to Int then we'll exit when it reaches 0 as we would
        # have completed one additional full chat iteration
        exit_after_next_group_chat: int | None = None

        if (self.mode != EndChatMode.MAX_ITERATIONS_ONLY) and any(
                [cur_task.llm_stops_by_itself for cur_task in self.tasks]) is False:
            raise IllogicalConfiguration(
                "You set an endChatMode in the GroupSolve() class that requires at least one Task() having the @llm_stops_by_itself argument set to True in its constructor. None of your tasks have it set to True.")

        if len(self.tasks) <= 1:
            raise IndexError("Must give at least 2 Tasks to make a group.")

        elif len(self.tasks) == 2:
            my_task1 = self.tasks[0]
            my_task2 = self.tasks[1]
            try:
                # First speaker
                my_task1.solve()

                # Giving prompt and AI output of first speaker to the second speaker
                if self.reconcile_first_message is True:
                    my_task2.agent.history.add(Message(MessageRole.USER, my_task1.task))
                    my_task2.agent.history.add(Message(MessageRole.ASSISTANT, my_task1.agent.history.get_last().content))
                # Second speaker (has the history of the first speaker)
                my_task2.solve()

                if self.reconcile_first_message is True:
                    my_task1.agent.history.add(Message(MessageRole.USER, my_task2.task))
                    my_task1.agent.history.add(Message(MessageRole.ASSISTANT, my_task2.agent.history.get_last().content))

                # (str, (TaskX, TaskY))
                last_generated_answer, tasks_run_order = self.set_shift_message()

                while self.max_iter > 0 and (exit_after_next_group_chat is None or exit_after_next_group_chat > 0):
                    logging.debug("Entering groupSolve.")

                    for current_task in tasks_run_order:
                        copy_task: Task = self._duplicate_task(current_task, last_generated_answer)
                        last_generated_answer: str = self._solve_copy(copy_task)
                        exit_after_next_chat, exit_after_next_group_chat = self._check_if_must_exit(exit_after_next_chat,
                                                                                                    exit_after_next_group_chat,
                                                                                                    copy_task)
                    logging.debug("One full iteration has been done (the two agents spoke)")
                    self.max_iter -= 1
                    logging.debug(f"Maximum iterations is now at {str(self.max_iter)}")
                    if isinstance(exit_after_next_group_chat, int):
                        exit_after_next_group_chat -= 1
            except ReachedTaskCompletion:
                pass

        elif len(self.tasks) >= 3:
            self._add_roleplay_prompts()

            for cur_task in self.tasks:
                # Changing the way the task is written to match the multi-agent format
                user_message: Message = Message(MessageRole.USER,
                                                f"[TaskManager]: {cur_task.agent.name}: this is your main task: `" + cur_task.task + "`")
                copy_task: Task = self._duplicate_task(cur_task, user_message.content)
                last_generated_answer: str = self._solve_copy(copy_task)
                self._reconcile_history(cur_task, user_message, last_generated_answer)
                exit_after_next_chat, exit_after_next_group_chat = self._check_if_must_exit(exit_after_next_chat,
                                                                                            exit_after_next_group_chat,
                                                                                            copy_task)

            try:
                while self.max_iter > 0 and (exit_after_next_group_chat is None or exit_after_next_group_chat > 0):
                    for cur_task in self.tasks:
                        user_message: Message = Message(MessageRole.USER,
                                                        f"[TaskManager]: {cur_task.agent.name} it's your turn to speak now.")
                        copy_task: Task = self._duplicate_task(cur_task, user_message.content)
                        last_generated_answer: str = self._solve_copy(copy_task)
                        self._reconcile_history(cur_task, user_message, last_generated_answer)
                        exit_after_next_chat, exit_after_next_group_chat = self._check_if_must_exit(
                            exit_after_next_chat, exit_after_next_group_chat, copy_task)
                    self.max_iter -= 1
                    if isinstance(exit_after_next_group_chat, int):
                        exit_after_next_group_chat -= 1
            except ReachedTaskCompletion:
                pass

    def _duplicate_task(self, task_to_duplicate: Task, last_generated_answer: str) -> Task:
        """
        Creates a new instance of an existing task. It will have the exact same configuration. However, we change the task prompt so it can solve something new.
        @param task_to_duplicate: Task : The Task to duplicate
        @param last_generated_answer: str : The prompt to change in the new Task
        @return: Task : The copy of the Task with the new prompt
        """
        copy_task2: Task = copy.deepcopy(task_to_duplicate)
        copy_task2.task = last_generated_answer
        copy_task2.agent = task_to_duplicate.agent
        return copy_task2

    def _solve_copy(self, copy_task: Task) -> str:
        last_generated_answer = copy_task.solve().content
        return last_generated_answer

    def _check_if_must_exit(self, exit_after_next_chat: bool, exit_after_next_group_chat: int | None, copy_task: Task) -> tuple[bool, int | None]:
        if exit_after_next_chat is True:
            raise ReachedTaskCompletion()

        if copy_task.llm_stops_by_itself is True:
            if copy_task.use_self_reflection is False:
                history_save = copy.deepcopy(copy_task.agent.history)
            # @copy_task is already a copy and won't be reused so no use to deepcopy it again.
            copy_task.task = ("[TaskManager]: " if len(self.tasks) >= 3 else "") + "In your opinion, what objectives from your initial task have you NOT completed ?"

            Task(copy_task.task, copy_task.agent).solve()
            if copy_task.use_self_reflection is True:
                history_save = copy.deepcopy(copy_task.agent.history)

            ms_temp_save: float = copy_task.agent.model_settings.temperature
            copy_task.agent.model_settings.temperature = 0.3
            answer: str = Task("To summarize in one word, did you still had some objectives to fulfill ? Answer ONLY by 'yes' or 'no'. Do not output anything else !", copy_task.agent, forget=True).solve().content

            copy_task.agent.model_settings.temperature = ms_temp_save
            copy_task.agent.history = history_save
            if "yes" in answer.lower():
                logging.debug(f"Still got objectives to fulfill ? SAID YES: {answer}")
            elif "no" in answer.lower():
                logging.debug(f"Still got objectives to fulfill ? SAID NO: {answer}")
                if self.mode == EndChatMode.END_CHAT_AFTER_FIRST_COMPLETION:
                    raise ReachedTaskCompletion()
                if self.mode == EndChatMode.ONE_LAST_CHAT_AFTER_FIRST_COMPLETION:
                    exit_after_next_chat = True
                if self.mode == EndChatMode.ONE_LAST_GROUP_CHAT_AFTER_FIRST_COMPLETION:
                    if exit_after_next_group_chat is None:
                        # This goes like 'None -> 2 -> 1 -> 0 -> if 0 then exit'. But I don't like the fact that the variable carries two pieces of information at a time. Checking if it's a None or an int and basing business logic on the result isn't great.
                        exit_after_next_group_chat = len(self.tasks)
                if self.mode == EndChatMode.ALL_TASK_MUST_COMPLETE:
                    self._tasks_completion_statuses[copy_task.uuid] = True
                    if all(self._tasks_completion_statuses.values()) is True:
                        raise ReachedTaskCompletion()
            else:
                logging.error("While checking if the task was completed Yacana couldn't determine if the LLM said Yes or No. As a decision must be made Yacana will assume it is a Yes. If you encounter this frequently please open an issue on the github. Thx !")
        return exit_after_next_chat, exit_after_next_group_chat

    def _reconcile_history(self, cur_task: Task, user_message: Message, last_generated_answer: str):
        # Adding task + answer to everyone except the one who just spoke in order to reconcile histories with all other agents
        for cur_task2 in self.tasks:
            if cur_task2.uuid != cur_task.uuid:
                cur_task2.agent.history.add(user_message)
                cur_task2.agent.history.add(Message(MessageRole.ASSISTANT, last_generated_answer))

    def _add_roleplay_prompts(self) -> None:
        for cur_task in self.tasks:
            other_speaker_names = ",".join(list(filter(None, [
                f"[{cur_task2.agent.name}]" if cur_task2.uuid != cur_task.uuid else "" for cur_task2 in self.tasks])))
            cur_task.agent.history.add(Message(MessageRole.USER,
                                                     f"[TaskManager]: You are entering a roleplay with multiple "
                                                     f"speakers where each one has his own objectives to fulfill. "
                                                     f"Each message must follow this syntax '[speaker_name]: "
                                                     f"message'.\n" + f"The other speakers are: "
                                                                      f"{other_speaker_names}.\nYour "
                                                                      f"speaker name is [{cur_task.agent.name}].\nI "
                                                                      f"will give you your task in the next message."))
            cur_task.agent.history.add(Message(MessageRole.ASSISTANT,
                                                     f"[{cur_task.agent.name}]: Received and acknowledged. I'm ready "
                                                     f"to execute my tasks as {cur_task.agent.name}. Please proceed "
                                                     f"with my assignment."))

