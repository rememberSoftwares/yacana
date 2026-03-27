import importlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Dict, List, Type, T

from .TokenCount import HFMessage, count_tokens_using_huggingface, count_tokens_using_tiktoken, count_tokens_using_regex
from .exceptions import SpecializedTokenCountingError
from .medias import Media


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
    arguments : str | dict
        The arguments to be passed to the tool. The Ollama agent wants a dict, while the OpenAI agent wants a string (JSON encoded).
    """

    def __init__(self, call_id: str, name: str, arguments: str | dict):
        self.call_id: str = call_id
        self.name: str = name
        self.arguments: str | dict = arguments

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
                "arguments": self.arguments
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


@unique
class MessageRole(str, Enum):
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
    tags : List[str]
        List of tags associated with the message.
    token_count : int | None
        Number of tokens in the message. None if not calculated yet.
    """

    _registry = {}

    def __init__(self, role: MessageRole, content: str | None = None, tool_calls: List[ToolCallFromLLM] | None = None, medias: List[str] | None = None, structured_output: Type[T] | None = None, tags: List[str] | None = None,
                 id: uuid.UUID | None = None) -> None:
        self.id = str(uuid.uuid4()) if id is None else str(id)
        self._role: MessageRole = role
        self._content: str | None = content
        self._tool_calls: List[ToolCallFromLLM] | None = tool_calls
        self._medias: List[str] = medias if medias is not None else []
        self.structured_output: Type[T] | None = structured_output
        self.tags: List[str] = list(tags) if tags is not None else []
        self.token_count: int | None = None

        # Checking that both @message and @tool_calls are neither None nor empty at the same time
        if content is None and (tool_calls is None or (tool_calls is not None and len(tool_calls) == 0)):
            raise ValueError("A Message must have a content or a tool call that is not None or [].")

    @property
    def role(self):
        return self._role

    @role.setter
    def role(self, value):
        self._role = value
        self.token_count = None

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value
        self.token_count = None

    @property
    def tool_calls(self):
        return self._tool_calls

    @tool_calls.setter
    def tool_calls(self, value):
        self._tool_calls = value
        self.token_count = None

    @property
    def medias(self):
        return self._medias

    @medias.setter
    def medias(self, value):
        self._medias = value
        self.token_count = None

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

        Returns
        -------
        Dict
            A dictionary representation of the message.
        """
        members = self.__dict__.copy()
        final_members: dict = {}

        members["type"] = self.__class__.__name__
        members["role"] = self.role.value
        if members["structured_output"] is not None:
            members["structured_output"] = self._structured_output_to_dict()
        if members["_tool_calls"] is not None:
            members["tool_calls"] = [tool_call._export() for tool_call in self.tool_calls]
        else:
            members["tool_calls"] = []

        for key, value in members.items():
            if key.startswith("_"):
                final_members[key[1:]] = value
            else:
                final_members[key] = value
        return final_members

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
        if "tool_calls" in members:
            members["tool_calls_tmp"] = [ToolCallFromLLM(tool_call["id"], tool_call["function"]["name"], tool_call["function"]["arguments"]) for tool_call in members["tool_calls"]]
            members["tool_calls"] = members["tool_calls_tmp"]
            del members["tool_calls_tmp"]
        else:
            members["tool_calls"] = []

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

    @abstractmethod
    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
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

    def get_token_count(self, llm_model_name: str = None, hugging_face_repo_name: str = None, hugging_face_token: str = None, padding_per_message: int = 4) -> int:
        """
        Get the total token count of messages in the history.

        * If the hugging_face_repo_name is provided, the token count will be calculated using the transformers library.
        (If the llm is gated (private), the hugging_face_token must be provided to access the repo.)
        * If the llm_model_name is provided and is an OpenAI LLM, the token count will be calculated using the tiktoken library.
        * If none of the above conditions are met, an approximative token count will be returned. This is only a rough estimate and should not be used for precise calculations.

        Note that using Tiktoken and transformers are precise but quite slow (loging to HF using the token is the worst). The approximative token count is very fast but not precise.

        Parameters
        ----------
        llm_model_name : str | None, optional
            The name of the LLM model. If provided and is an OpenAI model, the token count will be calculated using the tiktoken library.
        hugging_face_repo_name : str | None, optional
            The name of the Hugging Face repository for the model. If provided, the token count will be calculated using the transformers library.
        hugging_face_token : str | None, optional
            The Hugging Face access token for private models. Required if the model is gated (private).
        padding_per_message : int, optional
            The number of extra tokens to add per message for padding. Default is 4.

        Returns
        -------
        int
            The total token count of all messages in history.
        """
        if self.token_count:
            logging.debug("Using cache token count for message.")
            return self.token_count + padding_per_message

        simplified_message: HFMessage = self.get_message_as_hugging_face_dict()

        try:
            if hugging_face_repo_name:
                logging.debug("Will try to count tokens using Hugging Face.")
                self.token_count = count_tokens_using_huggingface([simplified_message], hugging_face_repo_name, hugging_face_token)
                return self.token_count + padding_per_message * len([simplified_message])
            elif llm_model_name:
                logging.debug("Will try to count tokens using Tiktoken.")
                self.token_count = count_tokens_using_tiktoken(llm_model_name, simplified_message)
                return self.token_count + padding_per_message
        except SpecializedTokenCountingError:
            pass  # Falling back to regex method

        logging.debug("Fallback : Will try to count tokens using regex (approximate).")
        self.token_count = count_tokens_using_regex(simplified_message)
        return self.token_count + padding_per_message

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

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        return {
            "role": self.role.value,
            "content": str(self.content)
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
        Mainly use to send the message to the inference server as JSON.

        Returns
        -------
        dict
            A dictionary containing the role, content, and media information.
        """
        message_as_dict = {
            "role": self.role.value,
            "content": self.content
        }
        if self.medias:
            message_as_dict["content"] = [
                {"type": "text", "text": self.content},
                *[Media.get_as_openai_dict(media_path) for media_path in self.medias]
            ]
        return message_as_dict

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        simplified_message: HFMessage = {
            "role": self.role.value,
            "content": str(self.content)
        }
        if self.medias:
            for media_path in self.medias:
                simplified_message["content"] += " " + Media.path_to_base64(media_path)
        return simplified_message

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
        super().__init__(role, content, tool_calls, None, structured_output, tags=tags, id=kwargs.get('id', None))

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

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        return {
            "role": self.role.value,
            "content": str(self.content)
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
        super().__init__(role, content, tool_calls, medias, structured_output, tags=tags, id=kwargs.get('id', None))

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

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        simplified_msg: HFMessage = {
            "role": self.role.value,
            "content": ""
        }
        if self.tool_calls:
            simplified_msg["content"] += " " + json.dumps([tool_call.get_tool_call_as_dict() for tool_call in self.tool_calls])
        return simplified_msg


class OpenAiToolCallingMessage(GenericMessage):
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

    Attributes
    ----------
    tool_call_id : str
        The ID of the tool call.
    """

    def __init__(self, content: str, tool_call_id: str, tags: List[str] = None, **kwargs):
        role = MessageRole.TOOL
        tool_calls = None
        medias = None
        structured_output = None
        self.tool_call_id = tool_call_id
        super().__init__(role, content, tool_calls, medias, structured_output, tags=tags, id=kwargs.get('id', None))

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
            **({"tool_call_id": self.tool_call_id} if self.tool_call_id is not None else {})
        }

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        simplified_msg: HFMessage = {
            "role": self.role.value,
            "content": str(self.content)
        }
        if self.tool_call_id:
            simplified_msg["content"] += " " + self.tool_call_id
        return simplified_msg


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
        super().__init__(role, content, tool_calls, medias, structured_output, tags=tags, id=kwargs.get('id', None))

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

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        return {
            "role": self.role.value,
            "content": str(self.content)
        }


class OllamaToolCallingMessage(GenericMessage):
    """
    Response from the LLM when a tool is called.

    Parameters
    ----------
    content : str
        The output of the tool.
    tool_call_name : str
        The ID of the tool call.
    tags : List[str], optional
        Optional list of tags associated with the message.
    **kwargs
        Additional keyword arguments passed to the parent class.

    Attributes
    ----------
    tool_call_name : str
        The name of the tool call.
    """

    def __init__(self, content: str, tool_call_name: str, tags: List[str] = None, **kwargs):
        role = MessageRole.TOOL
        tool_calls = None
        medias = None
        structured_output = None
        self.tool_call_name = tool_call_name
        super().__init__(role, content, tool_calls, medias, structured_output, tags=tags, id=kwargs.get('id', None))

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
            "name": self.tool_call_name
        }

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        simplified_msg: HFMessage = {
            "role": self.role.value,
            "content": str(self.content)
        }
        if self.tool_calls:
            simplified_msg["content"] += " " + self.tool_call_name
        return simplified_msg


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
        has_media: bool = False if self.medias is None or len(self.medias) == 0 else True
        return {
            "role": self.role.value,
            "content": self.content,
            **({"images": final_medias} if has_media else {}),
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

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        simplified_msg: HFMessage = {
            "role": self.role.value,
            "content": str(self.content)
        }
        if self.medias:
            final_medias = []
            for media in self.medias:
                final_medias.append(Media.path_to_base64(media))
            simplified_msg["content"] += " " + json.dumps(final_medias)
        return simplified_msg


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
            "content": self.content
        }

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        return {
            "role": self.role.value,
            "content": str(self.content)
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
        super().__init__(role, content, tool_calls, medias, structured_output, tags=tags, id=kwargs.get('id', None))

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

    def get_message_as_hugging_face_dict(self) -> HFMessage:
        """
        Convert the message to a dictionary format compatible with Hugging Face.
        This is used to count tokens using the Hugging Face tokenizers.
        The purpose of this method is to have 'content' contain every information like images, tool calls, etc.

        Returns
        -------
        HFMessage
            A dictionary representation of the message compatible with Hugging Face.
        """
        return {
            "role": self.role.value,
            "content": str(self.content)
        }
