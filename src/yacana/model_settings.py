from typing import List, Dict, Any
from abc import ABC


class ModelSettings(ABC):
    """
    Abstract base class for model settings.

    This class provides a base implementation for model settings with functionality
    for exporting settings, creating instances, and managing initial values.
    The constructor initializes the internal state for tracking initial values of settings. This means that you can update
    any setting and use .reset() to restore the initial values of all settings.
    """

    _registry = {}

    def __init__(self):
        self._initial_values = {}

    def __init_subclass__(cls, **kwargs):
        """
        Register a new subclass of ModelSettings.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        super().__init_subclass__(**kwargs)
        ModelSettings._registry[cls.__name__] = cls

    def _export(self) -> Dict:
        """
        Export the current settings to a dictionary.

        Returns
        -------
        Dict
            A dictionary containing all current settings, excluding internal
            tracking values. The dictionary includes a 'type' field indicating
            the class name.
        """
        members = self.__dict__.copy()
        members["type"] = self.__class__.__name__
        members.pop("_initial_values", None)
        return members

    @staticmethod
    def create_instance(members: Dict):
        """
        Create a new instance of a ModelSettings subclass from a dictionary.

        Parameters
        ----------
        members : Dict
            Dictionary containing the settings and the 'type' field indicating
            which subclass to instantiate.

        Returns
        -------
        ModelSettings
            A new instance of the appropriate ModelSettings subclass.

        Raises
        ------
        KeyError
            If the 'type' field is missing or the class is not registered.
        """
        cls_name = members.pop("type")
        cls = ModelSettings._registry.get(cls_name)
        return cls(**members)

    def get_settings(self) -> dict:
        """
        Get all current settings as a dictionary.

        Returns
        -------
        dict
            A dictionary containing all non-None settings that don't start with '_'.
        """
        return {key: value for key, value in self.__dict__.items() if value is not None and not key.startswith("_")}

    def reset(self) -> None:
        """
        Reset all settings to their initial values.

        This method restores all settings to the values they had when the instance
        was first created.

        Notes
        -----
        This method iterates through all initial values stored during initialization
        and restores them to their original state.
        """
        for key, value in self._initial_values.items():
            setattr(self, f"{key}", value)


class OllamaModelSettings(ModelSettings):
    """
    Settings for Ollama model configuration.

    This class encapsulates all settings recognized by the Ollama inference server.
    Note that some settings may have no effect when using other inference servers.

    Parameters
    ----------
    mirostat : int, optional
        Controls the model's creativity level (0: off, 1: on, 2: extra on).
    mirostat_eta : float, optional
        Adjusts how quickly the model learns from context (e.g., 0.1).
    mirostat_tau : float, optional
        Controls topic adherence (e.g., 5.0).
    num_ctx : int, optional
        Determines context window size (e.g., 4096).
    num_gqa : int, optional
        Controls parallel task handling (e.g., 8).
    num_gpu : int, optional
        Sets GPU utilization (e.g., 50).
    num_thread : int, optional
        Controls parallel processing (e.g., 8).
    repeat_last_n : int, optional
        Controls repetition prevention window (e.g., 64).
    repeat_penalty : float, optional
        Penalty for repeated content (e.g., 1.1).
    temperature : float, optional
        Controls response randomness (e.g., 0.7).
    seed : int, optional
        Random seed for reproducibility (e.g., 42).
    stop : List[str], optional
        Stop sequences for generation.
    tfs_z : float, optional
        Controls response randomness reduction (e.g., 2.0).
    num_predict : int, optional
        Maximum tokens to generate (e.g., 128).
    top_k : int, optional
        Limits token selection (e.g., 40).
    top_p : float, optional
        Controls token selection probability (e.g., 0.9).
    **kwargs
        Additional settings passed to the parent class.

    Attributes
    ----------
    mirostat : int, optional
        Controls the model's creativity level (0: off, 1: on, 2: extra on).
    mirostat_eta : float, optional
        Adjusts how quickly the model learns from context (e.g., 0.1).
    mirostat_tau : float, optional
        Controls topic adherence (e.g., 5.0).
    num_ctx : int, optional
        Determines context window size (e.g., 4096).
    num_gqa : int, optional
        Controls parallel task handling (e.g., 8).
    num_gpu : int, optional
        Sets GPU utilization (e.g., 50).
    num_thread : int, optional
        Controls parallel processing (e.g., 8).
    repeat_last_n : int, optional
        Controls repetition prevention window (e.g., 64).
    repeat_penalty : float, optional
        Penalty for repeated content (e.g., 1.1).
    temperature : float, optional
        Controls response randomness (e.g., 0.7).
    seed : int, optional
        Random seed for reproducibility (e.g., 42).
    stop : List[str], optional
        Stop sequences for generation.
    tfs_z : float, optional
        Controls response randomness reduction (e.g., 2.0).
    num_predict : int, optional
        Maximum tokens to generate (e.g., 128).
    top_k : int, optional
        Limits token selection (e.g., 40).
    top_p : float, optional
        Controls token selection probability (e.g., 0.9).
    """

    def __init__(self,
                 mirostat: int = None,
                 mirostat_eta: float = None,
                 mirostat_tau: float = None,
                 num_ctx: int = None,
                 num_gqa: int = None,
                 num_gpu: int = None,
                 num_thread: int = None,
                 repeat_last_n: int = None,
                 repeat_penalty: float = None,
                 temperature: float = None,
                 seed: int = None,
                 stop: List[str] = None,
                 tfs_z: float = None,
                 num_predict: int = None,
                 top_k: int = None,
                 top_p: float = None,
                 **kwargs) -> None:
        super().__init__()
        # Initialize all properties
        self.mirostat = mirostat
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.num_ctx = num_ctx
        self.num_gqa = num_gqa
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.repeat_last_n = repeat_last_n
        self.repeat_penalty = repeat_penalty
        self.temperature = temperature
        self.seed = seed
        self.stop = stop
        self.tfs_z = tfs_z
        self.num_predict = num_predict
        self.top_k = top_k
        self.top_p = top_p

        # Store the initial values for resetting
        self._initial_values = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}


class OpenAiModelSettings(ModelSettings):
    """
    Settings for OpenAI model configuration.

    This class encapsulates all settings recognized by the OpenAI API.
    Note that some settings may have no effect when using other inference servers.

    Parameters
    ----------
    audio : Any, optional
        Parameters for audio output when using audio modality.
    frequency_penalty : float, optional
        Penalty for token frequency (-2.0 to 2.0).
    logit_bias : Dict, optional
        Token bias adjustments (-100 to 100).
    logprobs : bool, optional
        Whether to return token log probabilities.
    max_completion_tokens : int, optional
        Maximum tokens to generate.
    metadata : Dict, optional
        Additional metadata (max 16 key-value pairs).
    modalities : List[str], optional
        Output types to generate (e.g., ["text", "audio"]).
    n : int, optional
        Number of completion choices to generate.
    prediction : Any, optional
        Configuration for predicted output.
    presence_penalty : float, optional
        Penalty for token presence (-2.0 to 2.0).
    reasoning_effort : str, optional
        Reasoning effort level ("low", "medium", "high").
    seed : int, optional
        Random seed for reproducibility.
    service_tier : str, optional
        Latency tier for processing ("auto" or "default").
    stop : str | List, optional
        Stop sequences for generation.
    store : bool, optional
        Whether to store completion output.
    stream_options : Any, optional
        Options for streaming response.
    temperature : float, optional
        Sampling temperature (0 to 2).
    top_logprobs : int, optional
        Number of top tokens to return (0 to 20).
    top_p : float, optional
        Nucleus sampling parameter.
    user : str, optional
        End-user identifier.
    web_search_options : Any, optional
        Web search configuration.
    **kwargs
        Additional settings passed to the parent class.

    Attributes
    ----------
    audio : Any, optional
        Parameters for audio output when using audio modality.
    frequency_penalty : float, optional
        Penalty for token frequency (-2.0 to 2.0).
    logit_bias : Dict, optional
        Token bias adjustments (-100 to 100).
    logprobs : bool, optional
        Whether to return token log probabilities.
    max_completion_tokens : int, optional
        Maximum tokens to generate.
    metadata : Dict, optional
        Additional metadata (max 16 key-value pairs).
    modalities : List[str], optional
        Output types to generate (e.g., ["text", "audio"]).
    n : int, optional
        Number of completion choices to generate.
    prediction : Any, optional
        Configuration for predicted output.
    presence_penalty : float, optional
        Penalty for token presence (-2.0 to 2.0).
    reasoning_effort : str, optional
        Reasoning effort level ("low", "medium", "high").
    seed : int, optional
        Random seed for reproducibility.
    service_tier : str, optional
        Latency tier for processing ("auto" or "default").
    stop : str | List, optional
        Stop sequences for generation.
    store : bool, optional
        Whether to store completion output.
    stream_options : Any, optional
        Options for streaming response.
    temperature : float, optional
        Sampling temperature (0 to 2).
    top_logprobs : int, optional
        Number of top tokens to return (0 to 20).
    top_p : float, optional
        Nucleus sampling parameter.
    user : str, optional
        End-user identifier.
    web_search_options : Any, optional
        Web search configuration.
    """

    def __init__(self,
                 audio: Any = None,
                 frequency_penalty: float = None,
                 logit_bias: Dict = None,
                 logprobs: bool = None,
                 max_completion_tokens: int = None,
                 metadata: Dict = None,
                 modalities: List[str] = None,
                 n: int = None,
                 prediction: Any = None,
                 presence_penalty: float = None,
                 reasoning_effort: str = None,
                 seed: int = None,
                 service_tier: str = None,
                 stop: str | List = None,
                 store: bool = None,
                 stream_options: Any = None,
                 temperature: float = None,
                 top_logprobs: int = None,
                 top_p: float = None,
                 user: str = None,
                 web_search_options: Any = None,
                 **kwargs) -> None:
        super().__init__()
        # Initialize all properties
        self.audio = audio
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.max_completion_tokens = max_completion_tokens
        self.metadata = metadata
        self.modalities = modalities
        self.n = n
        self.prediction = prediction
        self.presence_penalty = presence_penalty
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.service_tier = service_tier
        self.stop = stop
        self.store = store
        self.stream_options = stream_options
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.top_p = top_p
        self.user = user
        self.web_search_options = web_search_options


        # Store the initial values for resetting
        self._initial_values = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def reset(self) -> None:
        """
        Reset all settings to their initial values.

        This method restores all OpenAI model settings to the values they had when
        the instance was first created.

        Notes
        -----
        This method iterates through all initial values stored during initialization
        and restores them to their original state. This includes all OpenAI-specific
        parameters like temperature, top_p, and other generation settings.
        """
        for key, value in self._initial_values.items():
            setattr(self, f"{key}", value)

