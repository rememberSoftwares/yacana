import logging


class ColoredFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[96m'  # CYAN
    }
    RESET = '\033[0m'

    # Magenta '\033[95m'

    def format(self, record: logging.LogRecord):
        if record.levelname == "INFO" and record.msg.startswith("[PROMPT]"):
            log_color: str = '\033[92m'  # Green
        elif record.levelname == "INFO" and record.msg.startswith("[AI_RESPONSE]"):
            log_color: str = '\033[95m'  # Magenta
        else:
            log_color: str = self.COLORS.get(record.levelname, self.RESET)
        log_message: str = super().format(record)
        return f"\n{log_color}{log_message}{self.RESET}"


class LoggerManager:
    """Allows the user to set a correct level of debug. Note that this will impact the whole program.
    Defaults to INFO and will print most of the prompts and AI answers. Note that not all prompts are shown as some
    multi shot prompts are sometime injected and not displayed as they are set a posteriori and showing them at this
    exact moment would make the logs unlogical.

    Static methods
    ----------
    set_log_level(log_level: str | None)
    set_library_log_level(library_name: str, log_level: str)
    """
    LOG_LEVEL = "INFO"
    AVAILABLE_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", None]

    @staticmethod
    def setup_logging(log_level: str | None = None):
        """
        Initial setup logging level for the current program. You should not use this unless you know what you are doing. See set_log_level().
        :param log_level: str: One of => "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL" | None
        :return:
        """

        if log_level is None:
            log_level = LoggerManager.LOG_LEVEL
        else:
            log_level = log_level.upper()

        if log_level not in LoggerManager.AVAILABLE_LOG_LEVELS:
            log_level = "INFO"
            print("WARNING:root:Invalid value given to LOG_LEVEL. Defaulting to level: INFO")

        # Create and configure the handler with the ColoredFormatter
        handler = logging.StreamHandler()
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)

        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        root_logger.handlers = [handler]  # Replace default handlers with our colored handler

    @staticmethod
    def set_log_level(log_level: str | None):
        """
        Defines a log level to print messages to std. All levels have their specific colors. For information logs the
        color depends on if it is a user prompt or an AI answer. Setting @log_level to None will disable logging.

        @param log_level: str | None : Level of debug shown on std. One of => "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL" | None
        @return:
        """

        if log_level is None:
            # Disable logging if the log level is None
            logging.disable(logging.CRITICAL)
        elif log_level in LoggerManager.AVAILABLE_LOG_LEVELS:
            logging.getLogger().setLevel(getattr(logging, log_level))
            logging.disable(logging.NOTSET)  # Re-enable logging
        else:
            print(
                f"WARNING:root:Invalid value given to LOG_LEVEL. Keeping current level: {logging.getLevelName(logging.getLogger().level)}")

    @staticmethod
    def set_library_log_level(library_name: str, log_level: str):
        """
        Use to specify the log level of a specific python library as this setting is global to the whole program and can spam the output if all libraries start to log debug info.
        :param library_name: Name of the target libray
        :param log_level: Level of log
        :return:
        """
        if log_level in LoggerManager.AVAILABLE_LOG_LEVELS:
            library_logger = logging.getLogger(library_name)
            library_logger.setLevel(getattr(logging, log_level))
        else:
            print(f"WARNING:root:Invalid log level {log_level} for library {library_name}.")


# Automatically set up logging when the class is first imported
LoggerManager.setup_logging()
