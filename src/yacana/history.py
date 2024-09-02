import json
import uuid
from enum import Enum
from typing import List, Dict
from typing_extensions import Self


class MessageRole(Enum):
    """The available types of message creators.
    It can either be a message from the user or an answer from the LLM to the user's message.
    Developers can also set a system prompt that guides the LLM into a specific way of answering.

    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message:
    """The smallest entity representing an interaction with the LLM. Can either be a user message or an LLM message.
    Use the MessageRole() enum to specify who's message it is.

    Attributes
    ----------
    role : MessageRole
        From whom is the message from. See the MessageRole Enum
    content : str
        The actual message

    Methods
    ----------
    get_as_dict(self) -> Dict

    """

    def __init__(self, role: MessageRole, content: str) -> None:
        self.role: MessageRole = role
        self.content: str = content

    def get_as_dict(self) -> Dict:
        """
        Returns the alternation of messages that compose a conversation as a pure python dictionary
        @return:
        """
        return {'role': self.role.value, 'content': self.content}

    def __str__(self) -> str:
        result = {'role': self.role.value, 'content': self.content}
        return json.dumps(result)


class History:
    """Container for an alternation of Messages representing a conversation between the user and an LLM

    Attributes
    ----------

    Methods
    ----------
    add(message: Message) -> None
    get_as_dict() -> List[Dict]
    pretty_print() -> None
    create_check_point() -> str
    load_check_point(uid: str) -> None
    get_last() -> Message
    clean() -> None
    __str__() -> str

    """

    def __init__(self, messages: List[Message] = None) -> None:
        self._messages: List[Message] = [] if messages is None else messages
        # Looks like { "uid": { history_as_dict }, ... }
        self._checkpoints: dict = {}

    def add(self, message: Message) -> None:
        """
        Adds a new Message() to the list of messages composing the history of the conversation
        @param message: Message : A message with the sender and the message itself
        @return: None
        """
        self._messages.append(message)

    def get_as_dict(self) -> List[Dict]:
        """
        Returns the history (aka the conversation) as a pure python dictionary
        @return: The list of messages as a python dictionary
        """
        messages_as_dict: List[Dict] = []
        for message in self._messages:
            messages_as_dict.append(message.get_as_dict())
        return messages_as_dict

    def pretty_print(self) -> None:
        """
        Prints the history on the std with shinny colors
        @return: None
        """
        for message in self._messages:
            if message.role == MessageRole.USER:
                print('\033[92m[' + message.role.value + "]:\n" + message.content + '\033[0m')
            elif message.role == MessageRole.ASSISTANT:
                print('\033[95m[' + message.role.value + "]:\n" + message.content + '\033[0m')
            elif message.role == MessageRole.ASSISTANT:
                print('\033[96m[' + message.role.value + "]:\n" + message.content + '\033[0m')
            print("")

    def create_check_point(self) -> str:
        """
        Saves the current history so that you can load it back later. Useful when you want to keep a clean history in
        a flow that didn't worked out as expected and want to rollback.
        @return: str : A unique identifier that can be used to load the checkpoint at anytime.
        """
        uid: str = str(uuid.uuid4())
        self._checkpoints[uid] = self.get_as_dict()
        return uid

    def load_check_point(self, uid: str) -> None:
        """
        Loads the history saved from a particular checkpoint in time.
        It replaces the current history with the loaded one. Perfect for a timey wimey rollback in time.
        @param uid:
        @return:
        """
        self._messages = []
        self._load_as_dict(self._checkpoints[uid])

    def get_last(self) -> Message:
        """
        Returns the last message of the history. Not very useful but a good syntactic sugar to get the last item from
        the conversation
        @return: Message
        """
        if len(self._messages) <= 0:
            raise IndexError("History is empty")
        return self._messages[-1]

    def clean(self) -> None:
        """
        Resets the history, preserving only the initial system prompt
        @return: None
        """
        save_system_prompt: Message | None = None

        if len(self._messages) > 0:
            if self._messages[0].role == MessageRole.SYSTEM:
                save_system_prompt = self._messages[0]
            self._messages = []
            if save_system_prompt is not None:
                self._messages.append(save_system_prompt)

    def _load_as_dict(self, messages_dict: List[dict]) -> None:
        """
        !!Warning!! This is a concatenation of the given dict to the existing one.
        Loads a list of messages as raw List. ie: [
        {
            "role": "system",
            "content": "You are an AI assistant"
        },
        ...
        ]
        @param messages_dict: A python dictionary
        @return: None
        """
        for message_dict in messages_dict:
            matching_enum: MessageRole = next((role for role in MessageRole if role.value == message_dict["role"]),
                                              None)
            if matching_enum is None:
                raise ValueError("Invalid role during import")
            self._messages.append(Message(matching_enum, message_dict["content"]))
        pass

    def _concat_history(self, history: Self) -> None:
        """
        Concat a History instance to the current history.
        @param history:
        @return:
        """
        self._messages = self._messages + history._messages

    def __str__(self) -> str:
        result = []
        for message in self._messages:
            result.append(message.get_as_dict())
        return json.dumps(result)
