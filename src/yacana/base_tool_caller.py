from abc import abstractmethod, ABC
from typing import List, Type, Callable, Dict
from pydantic import BaseModel

from .tool import Tool


class BaseToolCaller(ABC):
    def __init__(self, agent):
        self.agent = agent  # full access to _chat(), config, tools, etc.

    @abstractmethod
    def propose_tool(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None, tags: List[str] | None = None):
        pass

    @abstractmethod
    def propose_tools(self, task: str, tools: List[Tool], json_output: bool, structured_output: Type[BaseModel] | None, medias: List[str] | None, streaming_callback: Callable | None = None, task_runtime_config: Dict | None = None, tags: List[str] | None = None):
        pass
