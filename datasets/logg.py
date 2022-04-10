
import logging

logger = logging.getLogger("energy difference prediction")
logger.setLevel(logging.INFO)

# Logging Format
FORMAT = "%(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)


def get_logger():
    """Get the logger of the plenpy package.

    See Also:
        The Python ``logging`` module.

    """
    return logging.getLogger("plenpy")


def enable():
    """Enable the plenpy logger. The logger is enabled by default.
    """
    lg = get_logger()
    lg.propagate = True
    return


def disable():
    """Disable the plenpy logger. The logger is enabled by default.
    """
    lg = get_logger()
    lg.propagate = False
    return


def set_level(level="info"):
    """Set the logging level.

    Args:
        level: Logging level. Available levels: 'info', 'debug', 'warning' and
            'critical'.

    """

    if level.lower() == "info":
        logger.setLevel(logging.INFO)

    elif level.lower() == "debug":
        logger.setLevel(logging.DEBUG)

    elif level.lower() == "warning":
        logger.setLevel(logging.WARNING)

    elif level.lower() == "critical":
        logger.setLevel(logging.CRITICAL)

    else:
        raise ValueError("Option '{}' is not valid.".format(level))

    return


def set_format(format="normal"):

    if format.lower() == "normal":
        format_str = "%(levelname)s: %(message)s"

    elif format.lower() == "detail":
        format_str = "%(levelname)s: [%(filename)s:%(lineno)3s - " \
                     "%(funcName)30s() ] %(message)s"

    elif format.lower() == "time":
        format_str = '%(asctime)-15s %(message)s'

    else:
        raise ValueError("Option '{}' is not valid.".format(format))

    logging.basicConfig(format=format_str)
    return
