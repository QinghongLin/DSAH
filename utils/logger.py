import os
from os.path import dirname, abspath
import logging

# pwd = dirname(dirname(abspath(__file__)))

def setup_logger(name_logfile, logs_dir, also_stdout=False):
    name_logfile = name_logfile.replace(';', '#')
    name_logfile = name_logfile.replace(':', '_')
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    fileHandler = logging.FileHandler(os.path.join(logs_dir, name_logfile), mode='w')
    fileHandler.setFormatter(formatter)
    if also_stdout:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    if also_stdout:
        logger.addHandler(streamHandler)
    return logger