from pathlib import Path
import logging

def make_logger(name, log_dir=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir/ f'{name}.log', 'a')
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt=datefmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger
