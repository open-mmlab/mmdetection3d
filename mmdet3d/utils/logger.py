import logging
from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO, name='mmdet'):
    """Get root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet".

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger depending on whether
            we are using 3D detector or 3D segmentor.
            Defaults to 'mmdet'.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)

    return logger
