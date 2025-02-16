import logging
import os
from datetime import datetime

#: The logger that will be used to log the execution information
execution_logger = logging.getLogger()


class LocalTimeFormatter(logging.Formatter):
    """Formatter to display time in local time instead of UTC."""
    
    def formatTime(self, record, datefmt=None):
        """Override the default time format to use local time."""
        # 获取当前时间的本地时间
        created_time = datetime.fromtimestamp(record.created)
        # 返回指定格式的时间字符串
        if datefmt:
            return created_time.strftime(datefmt)
        else:
            return created_time.isoformat()
        

def setup_logging(
    log_file=None,
    log_screen=True,
    datefmt="%m/%d/%Y %H:%M:%S %p",
    fmt="%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s",
    level=logging.INFO,
):
    """Setup the logging configuration.

    :param log_file: The log file path, defaults to None
    :type log_file: str, optional
    :param log_screen: Whether to log to the screen, defaults to True
    :type log_screen: bool, optional
    :param datefmt: The date format, defaults to "%m/%d/%Y %H:%M:%S %p"
    :type datefmt: str, optional
    :param fmt: The log format, defaults to "%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s"
    :type fmt: str, optional
    :param level: The log level, defaults to logging.INFO
    :type level: int, optional
    """
    execution_logger.handlers.clear()
    execution_logger.setLevel(level)

    log_formatter = LocalTimeFormatter(fmt=fmt, datefmt=datefmt)

    if log_screen:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        execution_logger.addHandler(console_handler)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        execution_logger.addHandler(file_handler)
