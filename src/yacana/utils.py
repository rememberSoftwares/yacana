import json
from enum import Enum


class Dotdict(dict):
    """dot.notation access to dictionary attributes with recursive conversion"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Dotdict(value)
            elif isinstance(value, list):
                self[key] = [Dotdict(item) if isinstance(item, dict) else item for item in value]

    def __getitem__(self, key):
        """Ensure dict-style access also returns Dotdict for nested dicts"""
        value = super().__getitem__(key)
        if isinstance(value, dict):
            return Dotdict(value)
        return value

    def model_dump_json(self, **kwargs):
        """Mimics the pydantic model_dump_json() method"""
        return json.dumps(self, **kwargs)


def to_base64(data):
    """Convert data to base64 string"""
    import base64
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')


class AgentType(Enum):
    OLLAMA = "OLLAMA"
    OPENAI = "OPENAI"
