import logging
import json
from pathlib import Path


def get_logger(name: str, save_dir: str = None) -> logging.Logger:
    """Function designed to get loggers for stdout and record to log file.

    Args:
        name: name of logger.
        save_dir: directory to save logs.

    Returns:
        logger with appropriate formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove previous handlers. Useful when logger is being redefined in the same run.
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Printing to stdout
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('{name} - {levelname} - {message}', style='{')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if save_dir:  # Save logs to a log file. Exclusive writing just in case of bugs.
        f_handler = logging.FileHandler(filename=str(save_dir) + '.log', mode='x')
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter('{asctime} - {name} - {levelname:8s} - {message}', style='{')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder designed to return the string of an object if it cannot be serialized.
    """
    def default(self, o):
        return str(o)


def save_dict_as_json(dict_data: dict, log_dir: str, save_name: str):
    file_dir = Path(log_dir, f'{save_name}.json')
    with open(file_dir, mode='x') as jf:  # Exclusive writing just in case of unimagined bugs.
        json.dump(dict_data, jf, indent=2, sort_keys=True, cls=CustomJSONEncoder)
