import logging, os

def get_logger(name: str, log_file=None):
    if log_file is None:
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'app.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
