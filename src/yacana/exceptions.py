class ToolError(Exception):
    """
    Exception raised when a tool encounters incorrect parameters.

    This exception is raised by the user from inside a tool when the tool determines
    that the given parameters are incorrect. The error message should be didactic
    to help the LLM fix its mistakes, otherwise it may loop until reaching
    MaxToolErrorIter.

    Parameters
    ----------
    message : str
        A descriptive message that will be given to the LLM to help fix the issue.
        Example: 'Parameter xxxx expected type int but got type string'.

    Notes
    -----
    The error message should be clear and instructive to help the LLM understand
    and correct the parameter issue.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class MaxToolErrorIter(Exception):
    """
    Exception raised when maximum error iterations are reached.

    This exception is raised when the maximum number of allowed errors has been
    reached. This includes both errors raised from the tool itself and errors
    when Yacana fails to call the tool correctly.

    Parameters
    ----------
    message : str
        Information about which specific iteration counter reached its maximum.

    Notes
    -----
    The error counters for ToolError and Yacana tool call errors are tracked
    separately. For example, if the maximum is set to 5, you could have 4 ToolError
    and 4 Yacana tool call errors (4 + 4 > 5) without raising this exception.
    The exception is only raised when one more error of either type occurs.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ReachedTaskCompletion(Exception):
    """
    Exception raised when a task has been successfully completed.

    This exception is used to signal that the current task has reached its
    completion state. It is typically used to break out of processing loops
    or to indicate successful task termination.
    """

    def __init__(self):
        pass


class IllogicalConfiguration(Exception):
    """
    Exception raised when the framework is used in an incoherent way.

    This exception indicates that the framework has been configured or used
    in a way that doesn't make logical sense. The stacktrace message should
    provide details about the specific configuration issue.

    Parameters
    ----------
    message : str
        A description of the illogical configuration or usage.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class TaskCompletionRefusal(Exception):
    """
    Exception raised when the model refuses to complete a task.

    This exception is raised when the model explicitly refuses to complete
    the requested task, typically due to ethical, safety, or capability reasons.

    Parameters
    ----------
    message : str
        The reason for the task completion refusal.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class UnknownResponseFromLLM(Exception):
    """
    Exception raised when the model returns an unknown response.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class McpBadToolConfig(Exception):
    """
    Exception raised when the input shema from the MCP server is not valid.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class McpBadTransport(Exception):
    """
    Exception raised when the protocol used by the MCP server response was neither application/JSON or
    SSE and parsing as JSON failed.

    Parameters
    ----------
    message : str
        A descriptive message about the transport issue.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class McpResponseError(Exception):
    """
    Exception raised when the MCP server sent an error as part of the response.

    Parameters
    ----------
    message : str
        A descriptive message about the transport issue.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class McpServerNotYetInitialized(Exception):
    """
    Exception raised when the MCP server is not yet initialized.

    This exception is used to indicate that the MCP server has not been set up
    or is not ready to handle requests.

    Parameters
    ----------
    message : str
        A descriptive message indicating that the MCP server is not initialized.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
