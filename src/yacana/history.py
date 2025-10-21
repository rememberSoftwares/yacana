import copy
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Dict, Type, T, Sequence, TypedDict
import importlib
import regex
from typing_extensions import Self
from abc import ABC, abstractmethod

from .TokenCount import HuggingFaceDetails
from .medias import Media


class HFMessage(TypedDict):
    role: str
    content: str


class MessageRole(Enum):
    """
    <b>ENUM:</b>  The available types of message creators.
    User messages are the ones that are sent by the user to the LLM.
    Assistant messages are the ones that are sent by the LLM to the user.
    System messages are the ones that defines the behavior of the LLM.
    Tool messages are the ones containing the result of a tool call and then sent to the LLM. Not all LLMs support this type of message.

    Attributes
    ----------
    USER : str
        User messages are the ones that are sent by the user to the LLM.
    ASSISTANT : str
        Assistant messages are the ones that are sent by the LLM to the user.
    SYSTEM : str
        System messages are the ones that defines the behavior of the LLM.
    TOOL : str
        Tool messages are the ones containing the result of a tool call and then sent to the LLM.
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class SlotPosition(Enum):
    """
    <b>ENUM:</b> The position of a slot in the history. This is only a syntactic sugar to make the code more readable.

    Attributes
    ----------
    BOTTOM : int
        The slot is at the bottom of the history.
    TOP : int
        The slot is at the top of the history.
    """
    BOTTOM = -1
    TOP = -2


class ToolCallFromLLM:
    """
    A simple object container for the tool call that is sent by the LLM to the user.

    Parameters
    ----------
    call_id : str
        The unique identifier for the tool call.
    name : str
        The name of the tool being called.
    arguments : dict
        The arguments to be passed to the tool.

    Attributes
    ----------
    call_id : str
        The unique identifier for the tool call.
    name : str
        The name of the tool being called.
    arguments : dict
        The arguments to be passed to the tool.
    """

    def __init__(self, call_id, name, arguments):
        self.call_id: str = call_id
        self.name: str = name
        self.arguments: dict = arguments

    def get_tool_call_as_dict(self):
        """
        Convert the tool call to a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the tool call information in the format:
            {
                "id": str,
                "type": "function",
                "function": {
                    "name": str,
                    "arguments": str
                }
            }
        """
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments)
            }
        }

    def _export(self):
        """
        Export the tool call as a dictionary. Mainly used for saving the object into a file.

        Returns
        -------
        dict
            A dictionary containing the tool call information in the format:
            {
                "id": str,
                "type": "function",
                "function": {
                    "name": str,
                    "arguments": dict
                }
            }
        """
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments
            }
        }


class GenericMessage(ABC):
    """
    Use for duck typing only.
    The smallest entity representing an interaction with the LLM.
    Use child class type to determine what type of message this is and the .role member to know from whom the message is from.

    Parameters
    ----------
    role : MessageRole
        From whom is the message from. See the MessageRole Enum.
    content : str | None, optional
        The actual message content. Can be None if tool_calls is provided.
    tool_calls : List[ToolCallFromLLM] | None, optional
        An optional list of tool calls that are sent by the LLM to the user.
    medias : List[str] | None, optional
        An optional list of path pointing to images or audio on the filesystem.
    structured_output : Type[T] | None, optional
        An optional pydantic model that can be used to store the result of a JSON response by the LLM.
    tool_call_id : str | None, optional
        The ID of the tool call this message is associated with.
    tags : List[str] | None, optional
        Optional list of tags associated with the message.
    id : uuid.UUID | None, optional
        The unique identifier of the message. If None, a new UUID will be generated.

    Attributes
    ----------
    id : str
        The unique identifier of the message.
    role : MessageRole
        From whom is the message from.
    content : str | None
        The actual message content.
    tool_calls : List[ToolCallFromLLM] | None
        List of tool calls associated with the message.
    medias : List[str]
        List of media file paths.
    structured_output : Type[T] | None
        Pydantic model for structured output.
    tool_call_id : str | None
        ID of the associated tool call.
    tags : List[str]
        List of tags associated with the message.
    """

    _registry = {}

    def __init__(self, role: MessageRole, content: str | None = None, tool_calls: List[ToolCallFromLLM] | None = None, medias: List[str] | None = None, structured_output: Type[T] | None = None, tool_call_id: str = None, tags: List[str] | None = None, id: uuid.UUID | None = None) -> None:
        self.id = str(uuid.uuid4()) if id is None else str(id)
        self._role: MessageRole = role
        self._content: str | None = content
        self._tool_calls: List[ToolCallFromLLM] | None = tool_calls
        self._medias: List[str] = medias if medias is not None else []
        self.structured_output: Type[T] | None = structured_output
        self.tool_call_id: str | None = tool_call_id
        self.tags: List[str] = list(tags) if tags is not None else []
        self._dirty = True

        # Checking that both @message and @tool_calls are neither None nor empty at the same time
        if content is None and (tool_calls is None or (tool_calls is not None and len(tool_calls) == 0)):
            raise ValueError("A Message must have a content or a tool call that is not None or [].")

    @property
    def role(self):
        return self._role

    @role.setter
    def role(self, value):
        self._role = value
        self._dirty = True

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value
        self._dirty = True

    @property
    def tool_calls(self):
        return self._tool_calls

    @tool_calls.setter
    def tool_calls(self, value):
        self._tool_calls = value
        self._dirty = True

    @property
    def medias(self):
        return self._medias

    @medias.setter
    def medias(self, value):
        self._medias = value
        self._dirty = True

    def __init_subclass__(cls, **kwargs):
        """
        Register a new subclass of GenericMessage.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        super().__init_subclass__(**kwargs)
        GenericMessage._registry[cls.__name__] = cls

    def _export(self) -> Dict:
        """
        Returns a pure python dictionary mainly to save the object into a file.
        None entries are omitted.

        Returns
        -------
        Dict
            A dictionary representation of the message with None entries omitted.
        """
        members = self.__dict__.copy()
        members["type"] = self.__class__.__name__
        members["role"] = self.role.value
        if members["structured_output"] is not None:
            members["structured_output"] = self._structured_output_to_dict()
        return members

    @staticmethod
    def create_instance(members: Dict):
        """
        Create a new instance of a GenericMessage subclass from a dictionary.

        Parameters
        ----------
        members : Dict
            Dictionary containing the message data.

        Returns
        -------
        GenericMessage
            A new instance of the appropriate GenericMessage subclass.
        """
        #  Converting the role string to its matching enum
        members["role"] = next((role for role in MessageRole if role.value == members["role"]), None)

        cls_name = members.pop("type")
        cls = GenericMessage._registry.get(cls_name)

        message = cls(**members)
        if message.structured_output is not None:
            message.structured_output = message._dict_to_structured_output(members["structured_output"])
        return message

    @abstractmethod
    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format.

        Returns
        -------
        dict
            A dictionary representation of the message.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented in the child class")

    def get_as_pretty(self) -> str:
        """
        Get a pretty-printed string representation of the message.

        Returns
        -------
        str
            The content of the message if available, otherwise a JSON string of tool calls.
        """
        if self.content is not None:
            return self.content
        else:
            return json.dumps([tool_call.get_tool_call_as_dict() for tool_call in self.tool_calls])

    def add_tags(self, tags: List[str]) -> None:
        """
        Add a tag to the message.

        Parameters
        ----------
        tags : List[str]
            The tag to add to the message.
        """
        self.tags.extend(tags)

    def remove_tag(self, tag: str) -> None:
        """
        Remove a tag from the message. Tags are used to filter messages in the history.

        Parameters
        ----------
        tag : str
            The tag to remove from the message.
        """
        if tag in self.tags:
            self.tags.remove(tag)

    def _structured_output_to_dict(self) -> Dict:
        """
        Convert the structured output to a dictionary.

        Returns
        -------
        Dict
            A dictionary representation of the structured output.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses when structured_output is not None.
        """
        raise NotImplemented(f"This method should be implemented in the child Message class when 'structured_output' is not None. Message type: {self.__class__.__name__}.")

    def _dict_to_structured_output(self, data: Dict):
        """
        Convert a dictionary to a structured output.

        Parameters
        ----------
        data : Dict
            The dictionary to convert to structured output.

        Returns
        -------
        Any
            The converted structured output.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses when structured_output is not None.
        """
        raise NotImplemented(f"This method should be implemented in the child Message class when 'structured_output' is not None. Message type: {self.__class__.__name__}.")

    def __str__(self) -> str:
        """
        Get a string representation of the message.

        Returns
        -------
        str
            A JSON string representation of the message.
        """
        return json.dumps(self._export())


class Message(GenericMessage):
    """
    For Yacana users or simple text based interactions.
    The smallest entity representing an interaction with the LLM. Can be manually added to the history.

    Parameters
    ----------
    role : MessageRole
        The role of the message sender.
    content : str
        The content of the message.
    tags : List[str], optional
        Optional list of tags associated with the message.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, **kwargs) -> None:
        super().__init__(role, content, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the role and content of the message.
        """
        return {
            "role": self.role.value,
            "content": self.content
        }


class OpenAIUserMessage(GenericMessage):
    """
    A message from the user to the LLM containing all features requested by the user.

    Parameters
    ----------
    role : MessageRole
        The role of the message sender.
    content : str
        The content of the message.
    tags : List[str], optional
        Optional list of tags associated with the message.
    medias : List[str], optional
        Optional list of media file paths.
    structured_output : Type[T], optional
        Optional pydantic model for structured output.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, medias: List[str] = None, structured_output: Type[T] = None, **kwargs):
        super().__init__(role, content, medias=medias, structured_output=structured_output, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format for OpenAI API.
        Mainly used to send the message to the inference server as JSON.

        Returns
        -------
        dict
            A dictionary containing the role, content, and media information.

        Raise
        -------
        ValueError
            If a media file type is not supported by OpenAI API.
        """
        message_as_dict = {
            "role": self.role.value,
            "content": self.content
        }
        if self.medias is not None:
            message_as_dict["content"] = [
                {"type": "text", "text": self.content},
                *[Media.get_as_openai_dict(media_path) for media_path in self.medias]
            ]
        return message_as_dict

    def _structured_output_to_dict(self) -> Dict:
        """
        Convert the structured output to a dictionary.
        Mainly used to export the pydantic model to a file.

        Returns
        -------
        Dict
            A dictionary containing the class information of the structured output.
        """
        return {
            "__class__": f"{self.structured_output.__module__}.{self.structured_output.__name__}"
        }

    def _dict_to_structured_output(self, data: Dict):
        """
        Convert a dictionary to a structured output.
        Mainly used to import the pydantic model from a file.

        Parameters
        ----------
        data : Dict
            The dictionary containing the class information.

        Returns
        -------
        Type[T]
            The structured output class.
        """
        full_class_path = data["__class__"]
        module_name, class_name = full_class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls


class OpenAITextMessage(GenericMessage):
    """
    Common message for OpenAI API. Mostly used for simple text messages.
    No special features like tool_calls, structured_output or media.

    Parameters
    ----------
    role : MessageRole
        The role of the message sender.
    content : str
        The content of the message.
    tags : List[str], optional
        Optional list of tags associated with the message.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """
    
    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, **kwargs):
        tool_calls = None
        structured_output = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, None, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the role and content of the message.
        """
        return {
            "role": self.role.value,
            "content": self.content
        }


class OpenAIFunctionCallingMessage(GenericMessage):
    """
    Response from OpenAI including tool calls to be parsed.

    Parameters
    ----------
    tool_calls : List[ToolCallFromLLM]
        List of tool calls to be parsed.
    tags : List[str], optional
        Optional list of tags associated with the message.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, tool_calls: List[ToolCallFromLLM], tags: List[str] = None, **kwargs):
        role = MessageRole.ASSISTANT
        content = None
        medias = None
        structured_output = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the role, content, and tool calls.
        """
        return {
            "role": self.role.value,
            "content": self.content,
            **({"tool_calls": [tool_call.get_tool_call_as_dict() for tool_call in self.tool_calls]} if self.tool_calls is not None else {})
        }


class OpenAIToolCallingMessage(GenericMessage):
    """
    Response from the LLM when a tool is called.

    Parameters
    ----------
    content : str
        The output of the tool.
    tool_call_id : str
        The ID of the tool call.
    tags : List[str], optional
        Optional list of tags associated with the message.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, content: str, tool_call_id: str, tags: List[str] = None, **kwargs):
        role = MessageRole.TOOL
        tool_calls = None
        medias = None
        structured_output = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the role, content, and tool call ID.
        """
        return {
            "role": self.role.value,
            "content": self.content,
            ** ({"tool_call_id": self.tool_call_id} if self.tool_call_id is not None else {}),
        }


class OpenAIStructuredOutputMessage(GenericMessage):
    """
    Response from OpenAI including structured output to be parsed.

    Parameters
    ----------
    role : MessageRole
        The role of the message sender.
    content : str
        The content of the message.
    structured_output : Type[T]
        The structured output to be parsed.
    tags : List[str], optional
        Optional list of tags associated with the message.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, role: MessageRole, content: str, structured_output: Type[T], tags: List[str] = None, **kwargs):
        tool_calls = None
        medias = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the role and content of the message.
        """
        return {
            "role": self.role.value,
            "content": self.content,
        }

    def _structured_output_to_dict(self) -> Dict:
        """
        Convert the structured output to a dictionary.

        Returns
        -------
        Dict
            A dictionary containing the class information and data of the structured output.
        """
        return {
            "__class__": f"{self.structured_output.__class__.__module__}.{self.structured_output.__class__.__name__}",
            "data": self.structured_output.model_dump()
        }

    def _dict_to_structured_output(self, data: Dict):
        """
        Convert a dictionary to a structured output.

        Parameters
        ----------
        data : Dict
            The dictionary containing the class information and data.

        Returns
        -------
        Type[T]
            The structured output instance.
        """
        full_class_path = data["__class__"]
        module_name, class_name = full_class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls(**data["data"])


class OllamaUserMessage(GenericMessage):
    """
    A message from the user to the LLM containing all features requested by the user (tools, medias, etc).

    Parameters
    ----------
    role : MessageRole
        The role of the message sender.
    content : str
        The content of the message.
    tags : List[str], optional
        Optional list of tags associated with the message.
    medias : List[str], optional
        Optional list of media file paths.
    structured_output : Type[T], optional
        Optional pydantic model for structured output.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, medias: List[str] = None, structured_output: Type[T] = None, **kwargs):
        super().__init__(role, content, medias=medias, structured_output=structured_output, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format for Ollama API.

        Returns
        -------
        dict
            A dictionary containing the role, content, and media information.
        """
        final_medias = []
        for media in self.medias:
            final_medias.append(Media.path_to_base64(media))
        return {
            "role": self.role.value,
            "content": self.content,
            **({"images": final_medias} if self.medias is not None else {}),
        }

    def _structured_output_to_dict(self) -> Dict:
        """
        Convert the structured output to a dictionary.

        Returns
        -------
        Dict
            A dictionary containing the class information of the structured output.
        """
        return {
            "__class__": f"{self.structured_output.__module__}.{self.structured_output.__name__}"
        }

    def _dict_to_structured_output(self, data: Dict):
        """
        Convert a dictionary to a structured output.

        Parameters
        ----------
        data : Dict
            The dictionary containing the class information.

        Returns
        -------
        Type[T]
            The structured output class.
        """
        full_class_path = data["__class__"]
        module_name, class_name = full_class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls


class OllamaTextMessage(GenericMessage):
    """
    Common message for Ollama. Mostly used for simple text messages.
    No special features like tool_calls, structured_output or media.

    Parameters
    ----------
    role : MessageRole
        The role of the message sender.
    content : str
        The content of the message.
    tags : List[str], optional
        Optional list of tags associated with the message.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, role: MessageRole, content: str, tags: List[str] = None, **kwargs):
        super().__init__(role, content, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the role, content, and media information.
        """
        return {
            "role": self.role.value,
            "content": self.content,
            **({"images": self.medias} if self.medias is not None else {}),
        }


class OllamaStructuredOutputMessage(GenericMessage):
    """
    Response from Ollama including structured output to be parsed.

    Parameters
    ----------
    role : MessageRole
        The role of the message sender.
    content : str
        The content of the message.
    structured_output : Type[T]
        The structured output to be parsed.
    tags : List[str], optional
        Optional list of tags associated with the message.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, role: MessageRole, content: str, structured_output: Type[T], tags: List[str] = None, **kwargs):
        tool_calls = None
        medias = None
        tool_call_id = None
        super().__init__(role, content, tool_calls, medias, structured_output, tool_call_id, tags=tags, id=kwargs.get('id', None))

    def get_message_as_dict(self):
        """
        Convert the message to a dictionary format.

        Returns
        -------
        dict
            A dictionary containing the role and content of the message.
        """
        return {
            "role": self.role.value,
            "content": self.content,
        }

    def _structured_output_to_dict(self) -> Dict:
        """
        Convert the structured output to a dictionary.

        Returns
        -------
        Dict
            A dictionary containing the class information and data of the structured output.
        """
        return {
            "__class__": f"{self.structured_output.__class__.__module__}.{self.structured_output.__class__.__name__}",
            "data": self.structured_output.model_dump()
        }

    def _dict_to_structured_output(self, data: Dict):
        """
        Convert a dictionary to a structured output.

        Parameters
        ----------
        data : Dict
            The dictionary containing the class information and data.

        Returns
        -------
        Type[T]
            The structured output instance.
        """
        full_class_path = data["__class__"]
        module_name, class_name = full_class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls(**data["data"])


class HistorySlot:
    """
    A slot is a container for messages. It can contain one or more messages.
    Most of the time it will only contain one message but when using `n=2` or`n=x` in the OpenAI API, it will contain multiple variations hence multiple messages.

    Parameters
    ----------
    messages : List[GenericMessage], optional
        A list of messages. Each message is a variation of the main message (defined by the
        @main_message_index parameter).
    raw_llm_json : str, optional
        The raw LLM JSON response for the slot. This is the raw JSON from the inference server.
        When using OpenAI this may contain more than one message hence the slot system acts as a
        container for the messages.
    **kwargs
        Additional keyword arguments including:
        id : str, optional
            The unique identifier for the slot.
        creation_time : int, optional
            The timestamp when the slot was created.

    Attributes
    ----------
    id : str
        The unique identifier for the slot.
    creation_time : int
        The timestamp when the slot was created.
    messages : List[GenericMessage]
        List of messages in the slot.
    raw_llm_json : str | None
        The raw LLM JSON response for the slot.
    main_message_index : int
        The index of the main message in the slot.
    """

    def __init__(self, messages: List[GenericMessage] = None, raw_llm_json: str = None, **kwargs):
        self.id = str(kwargs.get('id', uuid.uuid4()))
        self.creation_time: int = int(kwargs.get('creation_time', datetime.now().timestamp()))
        self.messages: List[GenericMessage] = [] if messages is None else messages
        self.raw_llm_json: str = raw_llm_json
        self.main_message_index: int = 0

    def set_main_message_index(self, message_index: int) -> None:
        """
        A slot can contain any number of concurrent message. But only one can be the main slot message and actually be part of the History.
        This method sets the index of the main message within the list of available messages in the slot.

        Parameters
        ----------
        message_index : int
            The index of the message to select as the main message.

        Raises
        ------
        IndexError
            If the message index is greater than the number of messages in the slot.
        """
        if message_index >= len(self.messages):
            raise IndexError("Index out of range: The message index is greater than the number of messages in the slot.")
        self.main_message_index = message_index

    def get_main_message_index(self) -> int:
        """
        Returns the index of the main message in the slot.

        Returns
        -------
        int
            The index of the currently selected message.
        """
        return self.main_message_index

    def add_message(self, message: GenericMessage):
        """
        Adds a new message to the slot.

        Parameters
        ----------
        message : GenericMessage
            The message to add to the slot.
        """
        self.messages.append(message)

    def get_message(self, message_index: int | None = None) -> GenericMessage:
        """
        Returns the main message of the slot or the one at the given index if index is provided.

        Parameters
        ----------
        message_index : int | None, optional
            The index of the message to return. If None, returns the currently selected message.

        Returns
        -------
        GenericMessage
            The requested message.

        Raises
        ------
        IndexError
            If the message index is greater than the number of messages in the slot.
        IllogicalConfiguration
            An HistorySlot should not be empty (no messages).
        """
        if message_index is None:
            return self.messages[self.main_message_index]
        if message_index >= len(self.messages):
            raise IndexError("Index out of range: The message index is greater than the number of messages in the slot.")
        else:
            return self.messages[message_index]

    def get_all_messages(self) -> List[GenericMessage]:
        """
        Returns all the messages in the slot.

        Returns
        -------
        List[GenericMessage]
            All messages in the slot.
        """
        return self.messages

    def set_raw_llm_json(self, raw_llm_json: str) -> None:
        """
        Sets the raw LLM JSON response for the slot.
        This is the raw JSON from the inference server. When using OpenAI this may contain more than one message hence the slot system acts as a container for the messages.

        Parameters
        ----------
        raw_llm_json : str
            The raw JSON response from the LLM.
        """
        self.raw_llm_json = raw_llm_json

    def _delete_message_by_index(self, message_index: int) -> None:
        """
        Deletes a message from the slot by index. An HistorySlot should NOT be empty (no messages). Use this method at your own risk.
        To delete a message from a slot, delete it using the `delete_message` method from the History class instead.
        This way, if the slot ends being empty it will be cleaned by the History class.

        Parameters
        ----------
        message_index : int
            The index of the message to delete.

        Raises
        ------
        IndexError
            If the message index is greater than the number of messages in the slot,
            or if trying to delete the last message in the slot.
        """
        if message_index >= len(self.messages):
            raise IndexError("Index out of range: The message index is greater than the number of messages in the slot.")
        
        if len(self.messages) <= 1:
            raise IndexError("Cannot delete the last message in a slot. Delete the slot from the history instead.")
        
        self.messages.pop(message_index)
        
        # Always reset to the first message for simplicity
        self.main_message_index = 0
        logging.debug("Main message index reset to 0 after message deletion")

    def _delete_message_by_id(self, message_id: str) -> None:
        """
        Deletes a message from the slot by id. An HistorySlot should NOT be empty (no messages). Use this method at your own risk.
        To delete a message from a slot, delete it using the `delete_message` method from the History class instead.
        This way, if the slot ends being empty it will be cleaned by the History class.

        Parameters
        ----------
        message_id : str
            The ID of the message to delete.

        Raises
        ------
        IndexError
            If trying to delete the last message in the slot.
        """
        for i, message in enumerate(self.messages):
            if message.id == message_id:
                self._delete_message_by_index(i)
                break

    def keep_only_selected_message(self):
        """
        Keeps only the currently selected message in the slot and deletes all the others.
        If there's only one message, this method does nothing.

        Raises
        ------
        IndexError
            If there are no messages in the slot.
        """
        if len(self.messages) == 0:
            raise IndexError("Cannot operate on an empty slot.")
            
        if len(self.messages) == 1:
            logging.debug("Keeping only selected message: Only one message in slot. Nothing to do.")
            return
            
        # Store the main message
        main_message = self.messages[self.main_message_index]
        
        # Clear all messages
        self.messages.clear()
        
        # Add back only the main message
        self.messages.append(main_message)
        self.main_message_index = 0
        logging.debug("Main message index reset to 0 after keeping only the selected message.")

    @staticmethod
    def create_instance(members: Dict):
        """
        Creates an instance of the HistorySlot class from a dictionary.
        Mainly used to import the object from a file.

        Parameters
        ----------
        members : Dict
            Dictionary containing the slot data.

        Returns
        -------
        HistorySlot
            A new instance of HistorySlot.
        """
        members["messages"] = [GenericMessage.create_instance(message) for message in members["messages"] if message is not None]
        return HistorySlot(**members)

    def _export(self) -> Dict:
        """
        Returns the slot as a dictionary.
        Mainly used to export the object to a file.

        Returns
        -------
        Dict
            A dictionary representation of the slot.
        """
        members = self.__dict__.copy()
        members["type"] = self.__class__.__name__
        members["messages"] = [message._export() for message in self.messages if message is not None]
        return members


class History:
    """
    Container for an alternation of Messages representing a conversation between the user and an LLM.
    To be precise, the history is a list of slots and not actual messages. Each slot contains at least one or more messages.
    This class does its best to hide the HistorySlot implementation. Meaning that many methods allows you to deal with the messages directly, but under the hood it always manages the slot wrapper.

    Parameters
    ----------
    llm_model_name : str | None, optional
        The name of the LLM model used for the conversation. Used to count tokens more accurately when using an OpenAi model. Will use Tiktoken under the hood.
    hugging_face_details: HuggingFaceDetails | None, optional
        Details for Hugging Face models, including repo name and access token. Used to count tokens more accurately when using an HuggingFace model. Will use the transformers library under the hood.
    **kwargs: Any
        Additional keyword arguments including:
        slots : List[HistorySlot], optional
            List of history slots.
        _checkpoints : Dict[str, list[HistorySlot]], optional
            Dictionary of checkpoints for the history.

    Attributes
    ----------
    slots : List[HistorySlot]
        List of history slots.
    _checkpoints : Dict[str, list[HistorySlot]]
        Dictionary of checkpoints for the history.
    llm_model_name : str | None
        The name of the LLM model used for the conversation.
    hugging_face_details: HuggingFaceDetails | None
        Details for Hugging Face models, including repo name and access token.
    """

    def __init__(self, llm_model_name: str | None = None, hugging_face_details: HuggingFaceDetails | None = None, **kwargs) -> None:
        self.slots: List[HistorySlot] = kwargs.get('slots', [])
        self._checkpoints: Dict[str, list[HistorySlot]] = kwargs.get('_checkpoints', {})
        self.llm_model_name = llm_model_name
        self.hugging_face_details: HuggingFaceDetails = hugging_face_details

    def add_slot(self, history_slot: HistorySlot, position: int | SlotPosition = SlotPosition.BOTTOM) -> None:
        """
        Adds a new slot to the history at the specified position.

        Parameters
        ----------
        history_slot : HistorySlot
            The slot to add to the history.
        position : int | SlotPosition, optional
            The position where to add the slot. Can be an integer or a SlotPosition enum value.
            Defaults to SlotPosition.BOTTOM.
        """
        if isinstance(position, SlotPosition):
            if position == SlotPosition.BOTTOM:
                self.slots.append(history_slot)
            elif position == SlotPosition.TOP:
                self.slots.insert(0, history_slot)
        else:
            self.slots.insert(position, history_slot)

    def get_last_slot(self) -> HistorySlot:
        """
        Returns the last slot of the history. A good syntactic sugar to get the last item from the conversation.

        Returns
        -------
        HistorySlot
            The last slot in the history.

        Raises
        ------
        IndexError
            If the history is empty.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[-1]

    def get_slot_by_index(self, index: int) -> HistorySlot:
        """
        Returns the slot at the given index.

        Parameters
        ----------
        index : int
            The index of the slot to return.

        Returns
        -------
        HistorySlot
            The slot at the given index.

        Raises
        ------
        IndexError
            If the history is empty or the index is out of range.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[index]

    def get_slot_by_id(self, id: str) -> HistorySlot:
        """
        Returns the slot with the given ID.

        Parameters
        ----------
        id : str
            The ID of the slot to return.

        Returns
        -------
        HistorySlot
            The slot with the given ID.

        Raises
        ------
        IndexError
            If the history is empty.
        ValueError
            If no slot with the given ID is found.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        for slot in self.slots:
            if slot.id == id:
                return slot
        raise ValueError(f"Slot with id {id} not found in history.")

    def get_slot_by_message(self, message: GenericMessage) -> HistorySlot:
        """
        Returns the slot containing the given message.

        Parameters
        ----------
        message : GenericMessage
            The message to search for.

        Returns
        -------
        HistorySlot
            The slot containing the message.

        Raises
        ------
        IndexError
            If the history is empty.
        ValueError
            If no slot contains the given message.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        for slot in self.slots:
            if any(message.id == msg.id for msg in slot.get_all_messages()):
                return slot
        raise ValueError(f"Message with id {message.id} not found in history.")

    def add_message(self, message: GenericMessage) -> HistorySlot:
        """
        Adds a new message to the history by creating a new slot containing one solo message inside.

        Parameters
        ----------
        message : GenericMessage
            The message to add to the history.

        Returns
        -------
        HistorySlot
            The new slot containing the message added to the history. Useful for chaining.
        """
        slot = HistorySlot([message])
        self.slots.append(slot)
        return slot

    def delete_slot(self, slot: HistorySlot) -> None:
        """
        Deletes a slot from the history.

        Parameters
        ----------
        slot : HistorySlot
            The slot to delete.

        Raises
        ------
        ValueError
            If the slot is not found in the history.
        """
        if slot not in self.slots:
            raise ValueError("Slot not found in history.")

        self.slots.remove(slot)
        logging.debug(f"Slot {slot} deleted from history.")

    def delete_slot_by_id(self, slot_id: str) -> None:
        """
        Deletes a slot from the history by its ID. If the ID does not exist, it logs a warning.

        Parameters
        ----------
        slot_id : str
            The ID of the slot to delete.
        """
        slot_to_delete = next((slot for slot in self.slots if slot.id == slot_id), None)
        if not slot_to_delete:
            logging.warning(f"No slot found with ID {slot_id}.")
        else:
            self.slots.remove(slot_to_delete)
            logging.debug(f"Slot with ID {slot_id} deleted from history.")

    def delete_message(self, message: Message) -> None:
        """
        Deletes a message from all slots in the history.

        Parameters
        ----------
        message : Message
            The message to delete.

        Raises
        ------
        ValueError
            If the message is not found in any slot.
        """
        for slot in self.slots:
            if message in slot.messages:
                if len(slot.messages) == 1:
                    self.delete_slot(slot)
                    logging.debug(f"Slot {slot} deleted from history because it contained only the message to delete.")
                else:
                    slot._delete_message_by_id(message.id)
                    logging.debug(f"Message {message} deleted from slot {slot}.")
                return
        raise ValueError("Message not found in any slot.")

    def delete_message_by_id(self, message_id: str) -> None:
        """
        Deletes a message from all slots in the history by its ID.

        Parameters
        ----------
        message_id : str
            The ID of the message to delete. If the ID does not exist, it logs a warning.
        """
        for slot in self.slots:
            message_to_delete = next((message for message in slot.messages if message.id == message_id), None)
            if message_to_delete:
                if len(slot.messages) == 1:
                    self.delete_slot(slot)
                    logging.debug(f"Slot {slot} deleted from history because it contained only the message to delete.")
                else:
                    slot._delete_message_by_id(message_to_delete.id)
                    logging.debug(f"Message with ID {message_id} deleted from slot {slot}.")
                return
        logging.warning(f"No message found with ID {message_id}.")

    def _export(self) -> Dict:
        """
        Returns the history as a dictionary.
        Mainly used to export the object to a file.

        Returns
        -------
        Dict
            A dictionary representation of the history.
        """
        members_as_dict = self.__dict__.copy()
        members_as_dict["_checkpoints"] = {}

        # Exporting checkpoints
        for uid, slots in self._checkpoints.items():
            exported_slots = [slot._export() for slot in slots]
            members_as_dict["_checkpoints"][uid] = exported_slots

        # Exporting slots
        slots_list: List[Dict] = []
        for slot in self.slots:
            slots_list.append(slot._export())
        members_as_dict["slots"] = slots_list
        return members_as_dict

    @staticmethod
    def create_instance(members: Dict):
        """
        Creates a new instance of History from a dictionary.

        Parameters
        ----------
        members : Dict
            Dictionary containing the history data.

        Returns
        -------
        History
            A new instance of History.
        """
        # Loading slots
        members["slots"] = [HistorySlot.create_instance(slot) for slot in members["slots"]]

        # Loading checkpoints
        for uid, slots in members["_checkpoints"].items():
            members["_checkpoints"][uid] = [HistorySlot.create_instance(slot) for slot in slots]
        return History(**members)

    def get_messages_as_dict(self) -> List[Dict]:
        """
        Returns all messages in the history as a list of dictionaries.

        Returns
        -------
        List[Dict]
            List of message dictionaries.
        """
        formated_messages = []
        for slot in self.slots:
            formated_messages.append(slot.get_message().get_message_as_dict())
        return formated_messages

    def pretty_print(self) -> None:
        """
        Prints the history to stdout with colored output.
        """
        for slot in self.slots:
            message = slot.get_message()
            if message.role == MessageRole.USER:
                print('\033[92m[' + message.role.value + "]:\n" + message.get_as_pretty() + '\033[0m')
            elif message.role == MessageRole.ASSISTANT:
                print('\033[95m[' + message.role.value + "]:\n" + message.get_as_pretty() + '\033[0m')
            elif message.role == MessageRole.SYSTEM:
                print('\033[93m[' + message.role.value + "]:\n" + message.get_as_pretty() + '\033[0m')
            elif message.role == MessageRole.TOOL:
                print('\033[96m[' + message.role.value + "]:\n" + message.get_as_pretty() + '\033[0m')
            print("")

    def create_check_point(self) -> str:
        """
        Creates a checkpoint of the current history state.

        Returns
        -------
        str
            A unique identifier for the checkpoint.
        """
        uid: str = str(uuid.uuid4())
        self._checkpoints[uid] = copy.deepcopy(self.slots)
        return uid

    def load_check_point(self, uid: str) -> None:
        """
        Loads a checkpoint of the history. Perfect for a timey wimey rollback in time.

        Parameters
        ----------
        uid : str
            The unique identifier of the checkpoint to load.
        """
        self.slots = self._checkpoints[uid]

    def get_message(self, index) -> GenericMessage:
        """
        Returns the message at the given index.

        Parameters
        ----------
        index : int
            The index of the message to return.

        Returns
        -------
        GenericMessage
            The message at the given index.

        Raises
        ------
        IndexError
            If the history is empty or the index is out of range.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[index].get_message()

    def get_messages_by_tags(self, tags: List[str], strict=False) -> Sequence[GenericMessage]:
        """
        Returns messages that match the given tags based on the matching mode.

        Parameters
        ----------
        tags : List[str]
            The tags to filter messages by.
        strict : bool, optional
            Controls the matching mode:
            - If False (default), returns messages that have ANY of the specified tags.
              For example, searching for ["tag1"] will match messages with ["tag1", "tag2"].
              This is useful for broad filtering.
            - If True, returns messages that have EXACTLY the specified tags (and possibly more).
              For example, searching for ["tag1", "tag2"] will match messages with ["tag1", "tag2", "tag3"]
              but not messages with just ["tag1"] or ["tag2"].
              This is useful for precise filtering.

        Returns
        -------
        Sequence[GenericMessage]
            List of messages matching the tag criteria.

        Raises
        ------
        IndexError
            If the history is empty.

        Examples
        --------
        >>> # Find all messages with tag1 (broad matching)
        >>> history.get_messages_by_tags(["tag1"])
        # Returns messages with ["tag1"], ["tag1", "tag2"], etc.

        >>> # Find messages with exactly tag1 and tag2 (strict matching)
        >>> history.get_messages_by_tags(["tag1", "tag2"], strict=True)
        # Returns messages with ["tag1", "tag2"], ["tag1", "tag2", "tag3"], etc.
        # But not messages with just ["tag1"] or ["tag2"]
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        messages = []
        for slot in self.slots:
            message_tags = set(slot.get_message().tags)
            if strict is False:
                # Non-strict mode: message must have ANY of the specified tags
                if set(tags).intersection(message_tags):
                    messages.append(slot.get_message())
            else:
                # Strict mode: message must have ALL specified tags
                if set(tags).issubset(message_tags):
                    messages.append(slot.get_message())


        return messages

    def get_last_message(self) -> GenericMessage:
        """
        Returns the last message in the history.

        Returns
        -------
        GenericMessage
            The last message in the history.

        Raises
        ------
        IndexError
            If the history is empty.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return self.slots[-1].get_message()

    def get_all_messages(self) -> List[Message]:
        """
        Returns all messages in the history.

        Returns
        -------
        List[Message]
            List of all messages in the history.

        Raises
        ------
        IndexError
            If the history is empty.
        """
        if len(self.slots) <= 0:
            raise IndexError("History is empty (no slots, so no messages)")
        return [slot.get_message() for slot in self.slots]

    def clean(self) -> None:
        """
        Resets the history, preserving only the initial system prompt if present.
        """
        if len(self.slots) > 0 and self.slots[0].get_message().role == MessageRole.SYSTEM:
            self.slots = [self.slots[0]]
        else:
            self.slots = []

    def _concat_history(self, history: Self) -> None:
        """
        Concatenates another history to this one.

        Parameters
        ----------
        history : Self
            The history to concatenate.
        """
        self.slots = self.slots + history.slots

    def get_token_count(self, padding_per_message: int = 4) -> int:
        """
        Get the total token count of messages in the history.

        * If the hugging face repo name is provided, the token count will be calculated using the transformers library.
        (If the llm is gated (private), the hugging_face_details.token must be provided to access the repo.)
        * If the llm_model_name is provided and is an OpenAI LLM, the token count will be calculated using the tiktoken library.
        * If none of the above conditions are met, an approximative token count will be returned. This is only a rough estimate and should not be used for precise calculations.

        Note that using Tiktoken and transformers are precise but quite slow (loging to HF using the token is the worst). The approximative token count is very fast but not precise.

        Returns
        -------
        int
            The token count of the message.
        """

        hf_formated_messages: List[HFMessage] = []

        # To apply chat template we need to simplify the representation of a message to only "role" and "content".
        for message in self.get_all_messages():
            history_message_as_dict: dict = message.get_message_as_dict()
            role: str = history_message_as_dict["role"]
            concat_of_all_dict_values: str = ""

            del history_message_as_dict["role"]

            for dict_value_item in history_message_as_dict.values():
                if isinstance(dict_value_item, list) and len(dict_value_item) == 0:
                    continue
                elif isinstance(dict_value_item, dict) and bool(dict_value_item) is False:
                    continue
                elif isinstance(dict_value_item, str):
                    concat_of_all_dict_values += dict_value_item + " "
                else:
                    concat_of_all_dict_values += json.dumps(dict_value_item) + " "

            concat_of_all_dict_values = concat_of_all_dict_values.rstrip()
            hf_formated_messages.append({"role": role, "content": concat_of_all_dict_values})
            print("chelou ce content = ", concat_of_all_dict_values)

        print("HF = ", json.dumps(hf_formated_messages))

        if self.hugging_face_details and self.hugging_face_details.repo_name:
            try:
                from transformers import AutoTokenizer
                if self.hugging_face_details.token:
                    from huggingface_hub import login
                    logging.debug("Logging into Hugging Face Hub to access private model for token counting. This may take some time...")
                    login(self.hugging_face_details.token)
                logging.debug("Loading tokenizer from Hugging Face Hub for model: " + self.hugging_face_details.repo_name)
                tokenizer = AutoTokenizer.from_pretrained(self.hugging_face_details.repo_name)
                tokens = tokenizer.apply_chat_template(hf_formated_messages, tokenize=True)
                print("Decoded tokens:", tokenizer.decode(tokens))
                return len(tokens) + padding_per_message * len(hf_formated_messages)
            except Exception as e:
                logging.warning(f"Could not load tokenizer for model {self.hugging_face_details.repo_name}. Falling back to approximative token count. Error: {e}")

        elif self.llm_model_name:
            import tiktoken
            token_count = 0
            try:
                logging.debug("Loading tiktoken encoding for model: " + self.llm_model_name)
                enc = tiktoken.encoding_for_model(self.llm_model_name)  # Getting correct encoding if it's an OpenAI model ONLY else ValueError
                for hf_message in hf_formated_messages:
                    token_count += len(enc.encode(hf_message["role"]) + enc.encode(hf_message["content"]))
                return token_count + padding_per_message * len(hf_formated_messages)
            except ValueError:
                logging.debug(f"Could not find encoding for model {self.llm_model_name}. This is normal if this model is not from OpenAI. You should set the @hugging_face_repo_nama and token in the Agent class so Yacana may use the correct tokeniser for this LLM. Falling back to approximative token count instead.")

        token_count = 0
        token_match_regex = regex.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", regex.MULTILINE)
        logging.debug("Using approximative token count method.")
        for hf_message in hf_formated_messages:
            print("len de ", hf_message["role"], "et len de ", hf_message["content"])
            print(str(len(token_match_regex.findall(hf_message["role"]))) + str(len(token_match_regex.findall(hf_message["content"]))))
            token_count += len(token_match_regex.findall(hf_message["role"]) + token_match_regex.findall(hf_message["content"]))
        print("final padding = ", padding_per_message * len(hf_formated_messages))
        return token_count + padding_per_message * len(hf_formated_messages)

    def __str__(self) -> str:
        """
        Returns a string representation of the history.

        Returns
        -------
        str
            A JSON string representation of the history.
        """
        result = []
        for slot in self.slots:
            result.append(slot.get_message()._export())
        return json.dumps(result)
