class ToolError(Exception):
    """Raised by the user from inside a tool when the tool 'thinks' that the given parameters are incorrect.
    The message given in the raise(...) should help the LLM to fix its mistakes or else it will probably loop till
    reaching MaxToolErrorIter. The error message must be didactic.
    For instance: 'Parameter xxxx expected type int but got type string'.

    """
    def __init__(self, message):
        """
        Raised by the user from inside a tool when the tool 'thinks' that the given parameters are incorrect.
        :param message: str: The message will be given to the LLM, so it tries to fix the issue.
        """
        self.message = message
        super().__init__(self.message)


class MaxToolErrorIter(Exception):
    """Raised when the maximum amount of errors has been reached. This counts errors raised from the tool but also when
    Yacana fails to all the tool correctly. However, note that both situation have their own counter variable. Meaning
    that if set to 5 you can have 4 ToolError and 4 Yacana tool call error ((4 + 4) > 5) and it will be okay. It is when
    we reach one more error of either types that this exception will be raised and chat will stop.

    """
    def __init__(self, message):
        """
        Raised when the maximum amount of errors has been reached.
        :param message: Info on what specific iteration counter got maxed out
        """
        self.message = message
        super().__init__(self.message)


class ReachedTaskCompletion(Exception):

    def __init__(self):
        pass


class IllogicalConfiguration(Exception):
    """You used the framework in an incoherent way.
    Seek the stacktrace message.

    """
    def __init__(self, message):
        """
        You used the framework in an incoherent way. Seek the stacktrace message.
        :param message:
        """
        self.message = message
        super().__init__(self.message)