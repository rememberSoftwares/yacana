class ModelSettings:
    """Class to encapsulate the settings available into the inference server.
    Note that these are all recognised by Ollama but may have no effect when using other inference servers.
    
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
                 stop: str = None,
                 tfs_z: float = None,
                 num_predict: int = None,
                 top_k: int = None,
                 top_p: float = None) -> None:
        """
        @param mirostat: Like a volume control for the machine’s “creativity.” (Example: 0 is off, 1 is on, 2 is extra on)
        @param mirostat_eta: Adjusts how quickly the machine learns from what it’s currently talking about. (Example: 0.1)
        @param mirostat_tau: Helps decide if the machine should stick closely to the topic. (Example: 5.0)
        @param num_ctx: Determines how much of the previous conversation the machine can remember at once. (Example: 4096)
        @param num_gqa: Like tuning how many different tasks the machine can juggle at once. (Example: 8)
        @param num_gpu: Sets how many “brains” (or parts of the computer’s graphics card) the machine uses. (Example: 50)
        @param num_thread: Determines how many separate conversations or tasks the machine can handle at the same time. (Example: 8)
        @param repeat_last_n: How much of the last part of the conversation to try not to repeat. (Example: 64)
        @param repeat_penalty: A nudge to encourage the machine to come up with something new if it starts repeating itself. (Example: 1.1)
        @param temperature: Controls how “wild” or “safe” the machine’s responses are. (Example: 0.7)
        @param seed: Sets up a starting point for generating responses. (Example: 42)
        @param stop: Tells the machine when to stop talking, based on certain cues or keywords. (Example: "AI assistant:")
        @param tfs_z: Aims to reduce randomness in the machine’s responses. (Example: 2.0)
        @param num_predict: Limits how much the machine can say in one go. (Example: 128)
        @param top_k: Limits the machine’s word choices to the top contenders. (Example: 40)
        @param top_p: Works with top_k to fine-tune the variety of the machine’s responses. (Example: 0.9)
        """
        # Initialize all properties
        self._mirostat = mirostat
        self._mirostat_eta = mirostat_eta
        self._mirostat_tau = mirostat_tau
        self._num_ctx = num_ctx
        self._num_gqa = num_gqa
        self._num_gpu = num_gpu
        self._num_thread = num_thread
        self._repeat_last_n = repeat_last_n
        self._repeat_penalty = repeat_penalty
        self._temperature = temperature
        self._seed = seed
        self._stop = stop
        self._tfs_z = tfs_z
        self._num_predict = num_predict
        self._top_k = top_k
        self._top_p = top_p

        # Store the initial values for resetting
        self._initial_values = {
            'mirostat': mirostat,
            'mirostat_eta': mirostat_eta,
            'mirostat_tau': mirostat_tau,
            'num_ctx': num_ctx,
            'num_gqa': num_gqa,
            'num_gpu': num_gpu,
            'num_thread': num_thread,
            'repeat_last_n': repeat_last_n,
            'repeat_penalty': repeat_penalty,
            'temperature': temperature,
            'seed': seed,
            'stop': stop,
            'tfs_z': tfs_z,
            'num_predict': num_predict,
            'top_k': top_k,
            'top_p': top_p,
        }

    # Getter and setter for mirostat
    @property
    def mirostat(self) -> int:
        return self._mirostat

    @mirostat.setter
    def mirostat(self, value: int) -> None:
        self._mirostat = value

    # Getter and setter for mirostat_eta
    @property
    def mirostat_eta(self) -> float:
        return self._mirostat_eta

    @mirostat_eta.setter
    def mirostat_eta(self, value: float) -> None:
        self._mirostat_eta = value

    # Getter and setter for mirostat_tau
    @property
    def mirostat_tau(self) -> float:
        return self._mirostat_tau

    @mirostat_tau.setter
    def mirostat_tau(self, value: float) -> None:
        self._mirostat_tau = value

    # Getter and setter for num_ctx
    @property
    def num_ctx(self) -> int:
        return self._num_ctx

    @num_ctx.setter
    def num_ctx(self, value: int) -> None:
        self._num_ctx = value

    # Getter and setter for num_gqa
    @property
    def num_gqa(self) -> int:
        return self._num_gqa

    @num_gqa.setter
    def num_gqa(self, value: int) -> None:
        self._num_gqa = value

    # Getter and setter for num_gpu
    @property
    def num_gpu(self) -> int:
        return self._num_gpu

    @num_gpu.setter
    def num_gpu(self, value: int) -> None:
        self._num_gpu = value

    # Getter and setter for num_thread
    @property
    def num_thread(self) -> int:
        return self._num_thread

    @num_thread.setter
    def num_thread(self, value: int) -> None:
        self._num_thread = value

    # Getter and setter for repeat_last_n
    @property
    def repeat_last_n(self) -> int:
        return self._repeat_last_n

    @repeat_last_n.setter
    def repeat_last_n(self, value: int) -> None:
        self._repeat_last_n = value

    # Getter and setter for repeat_penalty
    @property
    def repeat_penalty(self) -> float:
        return self._repeat_penalty

    @repeat_penalty.setter
    def repeat_penalty(self, value: float) -> None:
        self._repeat_penalty = value

    # Getter and setter for temperature
    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = value

    # Getter and setter for seed
    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    # Getter and setter for stop
    @property
    def stop(self) -> str:
        return self._stop

    @stop.setter
    def stop(self, value: str) -> None:
        self._stop = value

    # Getter and setter for tfs_z
    @property
    def tfs_z(self) -> float:
        return self._tfs_z

    @tfs_z.setter
    def tfs_z(self, value: float) -> None:
        self._tfs_z = value

    # Getter and setter for num_predict
    @property
    def num_predict(self) -> int:
        return self._num_predict

    @num_predict.setter
    def num_predict(self, value: int) -> None:
        self._num_predict = value

    # Getter and setter for top_k
    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._top_k = value

    # Getter and setter for top_p
    @property
    def top_p(self) -> float:
        return self._top_p

    @top_p.setter
    def top_p(self, value: float) -> None:
        self._top_p = value

    def reset(self) -> None:
        """
        Reset all properties to their initial values
        @return: None
        """
        for key, value in self._initial_values.items():
            setattr(self, f"_{key}", value)

    def get_settings(self) -> dict:
        """
        Returns a dictionary of all the settings and their current values, excluding None values.

        Returns
        -------
        dict
            A dictionary containing the machine settings that have been set.
        """
        settings = {
            "mirostat": self.mirostat,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_tau": self.mirostat_tau,
            "num_ctx": self.num_ctx,
            "num_gqa": self.num_gqa,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "seed": self.seed,
            "stop": self.stop,
            "tfs_z": self.tfs_z,
            "num_predict": self.num_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        return {key: value for key, value in settings.items() if value is not None}