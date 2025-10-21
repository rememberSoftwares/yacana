import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Type, T, Callable, Dict
from pydantic import BaseModel

from .TokenCount import HuggingFaceDetails
from .history import History, GenericMessage, MessageRole, Message
from .logging_config import LoggerManager
from .model_settings import ModelSettings
from .tool import Tool
from .exceptions import IllogicalConfiguration

logger = logging.getLogger(__name__)


class GenericAgent(ABC):
    """
    Representation of an LLM. This class gives ways to interact with the LLM that is being assign to it.
    However, an agent should not be controlled directly but assigned to a Task(). When the task is marked as solved
    then the agent will interact with the prompt inside the task and output an answer. This class is more about
    configuring the agent than interacting with it.

    Parameters
    ----------
    name : str
        Name of the agent. Can be used during conversations. Use something short and meaningful that doesn't contradict the system prompt.
    model_name : str
        Name of the LLM model that will be sent to the inference server. For instance 'llama:3.1" or 'mistral:latest' etc.
    model_settings : ModelSettings
        All settings that Ollama currently supports as model configuration. This needs to be tested with other inference servers.
        This allows modifying deep behavioral patterns of the LLM.
    system_prompt : str | None, optional
        Defines the way the LLM will behave. For instance set the SP to "You are a pirate" to have it talk like a pirate.
    endpoint : str | None, optional
        By default will look for Ollama endpoint on your localhost. If you are using a VM with GPU then update this to the remote URL + port.
    api_token : str, optional
        The API token used for authentication with the inference server.
    headers : dict, optional
        Custom headers to be sent with the inference request. If None, an empty dictionary will be used.
    runtime_config : Dict | None, optional
        Runtime configuration for the agent.
    history : History | None, optional
        The conversation history. If None, a new History instance will be created.
    task_runtime_config : Dict | None, optional
        Runtime configuration for tasks.
    thinking_tokens : Tuple[str, str] | None, optional
        A tuple containing the start and end tokens of a thinking LLM. For instance, "<think>" and "</think>" for Deepseek-R1.
        Setting this prevents the framework from getting sidetracked during the thinking steps and helps maintain focus on the final result.

    Raises
    ------
    ValueError
        If model_settings is None.

    Attributes
    ----------
    name : str
        Name of the agent.
    model_name : str
        Name of the LLM model.
    system_prompt : str | None
        System prompt defining the LLM's behavior.
    model_settings : ModelSettings
        Model configuration settings.
    api_token : str
        API token for authentication.
    headers : dict
        Custom headers for requests.
    endpoint : str | None
        Endpoint URL for the inference server.
    runtime_config : Dict
        Runtime configuration for the agent.
    task_runtime_config : Dict
        Runtime configuration for tasks.
    history : History
        The conversation history.
    _tags : List[str]
        Internal list of tags.
    thinking_tokens : Tuple[str, str] | None
        A tuple containing the start and end tokens of a thinking LLM. For instance, "<think>" and "</think>" for Deepseek-R1.
        Setting this prevents the framework from getting sidetracked during the thinking steps and helps maintain focus on the final result.
    """

    _registry = {}

    def __init__(self, name: str, model_name: str, model_settings: ModelSettings, system_prompt: str | None = None, endpoint: str | None = None,
                 api_token: str = "", headers=None, runtime_config: Dict | None = None, history: History | None = None, task_runtime_config: Dict | None = None,
                 thinking_tokens: tuple[str, str] | None = None, hugging_face_repo_name: str | None = None, hugging_face_token: str | None = None) -> None:
        if model_settings is None:
            raise ValueError("model_settings cannot be None. Please provide a valid ModelSettings instance.")

        # Vérification de la validité de thinking_tokens
        if thinking_tokens is not None:
            if (not isinstance(thinking_tokens, tuple) or len(thinking_tokens) != 2 or
                    not all(isinstance(t, str) and t.strip() for t in thinking_tokens)):
                raise IllogicalConfiguration("Thinking tokens must be a tuple of two strings representing the start and end tags outputted by the LLM when thinking.")

        self.name: str = name
        self.model_name: str = model_name
        self.system_prompt: str | None = system_prompt
        self.model_settings: ModelSettings = model_settings
        self.api_token: str = api_token
        self.headers = {} if headers is None else headers
        self.endpoint: str | None = endpoint
        self.runtime_config = runtime_config if runtime_config is not None else {}
        self.task_runtime_config = task_runtime_config if task_runtime_config is not None else {}
        self._tags: List[str] = []
        self.thinking_tokens: tuple[str, str] | None = thinking_tokens
        self.hugging_face_repo_name: str | None = hugging_face_repo_name
        self.hugging_face_token: str | None = hugging_face_token

        # Configuring History to use the correct model name and HF repo specific to this agent (These are not mandatory but if provided will allow to get an accurate token count)
        self.history: History = history if history is not None else History()
        self.history.llm_model_name = model_name
        self.history.hugging_face_details = HuggingFaceDetails(hugging_face_repo_name, hugging_face_token)

        if self.system_prompt is not None and history is None:
            self.history.add_message(Message(MessageRole.SYSTEM, system_prompt))

    def __init_subclass__(cls, **kwargs):
        """
        Register a new subclass of GenericAgent.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        super().__init_subclass__(**kwargs)
        GenericAgent._registry[cls.__name__] = cls

    def _strip_thinking_tags(self, message: str) -> str:
        """
        Strips tags from a string.

        Parameters
        ----------
        message : str
            The message to strip tags from.

        Returns
        -------
        str
            The string without tags.
        """
        if self.thinking_tokens is None or len(self.thinking_tokens) != 2:
            return message
        pattern = fr'{self.thinking_tokens[0]}.*?{self.thinking_tokens[1]}'
        return re.sub(pattern, '', message, flags=re.DOTALL)

    def export_to_file(self, file_path: str, strip_api_token=False, strip_headers=False) -> None:
        """
        Exports the current agent configuration to a file.

        This contains all the agents data and history. This means that you can use the
        import_from_file method to load this agent back again and continue where you left off.

        Warning
        -------
        This will leak API keys and headers unless strip_api_token and strip_headers are set to True.

        Parameters
        ----------
        file_path : str
            Path of the file in which to save the data. Specify the path + filename.
            Be wary when using relative paths.
        strip_api_token : bool, optional
            If True, removes the API token from the exported data. Defaults to False.
        strip_headers : bool, optional
            If True, removes headers from the exported data. Defaults to False.
        """
        members_as_dict = self.__dict__.copy()
        members_as_dict = {k: v for k, v in members_as_dict.items() if not k.startswith('_')}
        members_as_dict["type"] = self.__class__.__name__
        members_as_dict["model_settings"] = self.model_settings._export()
        members_as_dict["history"] = self.history._export()

        if self.api_token is not None and self.api_token != "" and strip_api_token is False:
            logging.warning("Saving the agent state will leak the API key to the destination file. Consider using @strip_api_token=True to remove it.")
        if self.headers is not None and bool(self.headers) is not False and strip_headers is False:
            logging.warning("Saving the agent state will leak headers content to the destination file. Consider using @strip_headers=True to remove it.")
        if strip_api_token:
            members_as_dict["api_token"] = None
        if strip_headers:
            members_as_dict["headers"] = {}

        with open(file_path, 'w') as file:
            json.dump(members_as_dict, file, indent=4)
        logging.info("Agent state successfully exported to %s", file_path)

    @classmethod
    def import_from_file(cls, file_path: str) -> 'GenericAgent':
        """
        Loads the state previously exported from the export_to_file method.

        This will return an Agent in the same state as it was before it was saved,
        allowing you to resume the agent conversation even after the program has exited.

        Parameters
        ----------
        file_path : str
            The path to the file from which to load the Agent.

        Returns
        -------
        GenericAgent
            A newly created Agent that is a copy from disk of a previously exported agent.
        """
        with open(file_path, 'r') as file:
            members: Dict = json.load(file)

        cls_name = members.pop("type")
        members["model_settings"] = ModelSettings.create_instance(members["model_settings"])
        members["history"] = History.create_instance(members["history"])
        cls = GenericAgent._registry.get(cls_name)
        return cls(**members)

    @abstractmethod
    def _interact(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None, task_runtime_config: Dict | None, tags: List[str] | None) -> GenericMessage:
        """
        Abstract method to start the inference using given parameters.

        Parameters
        ----------
        task : str
            The task to perform.
        tools : List[Tool]
            List of tools available for the agent.
        json_output : bool
            Whether to output JSON.
        structured_output : Type[BaseModel] | None
            Optional structured output type.
        medias : List[str] | None
            Optional list of media files.
        streaming_callback : Callable | None
            Optional callback for streaming responses.
        task_runtime_config : Dict | None
            Optional runtime configuration for the task.
        tags : List[str] | None
            Optional list of tags.
        Returns
        -------
        GenericMessage
            The response message from the agent.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplemented(f"This method must be subclassed by the child class. It starts the inference using given parameters.")
