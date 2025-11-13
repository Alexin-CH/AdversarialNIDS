import logging
import os
from datetime import datetime

class LoggerManager:
    def __init__(self, log_dir="logs", log_name="train_log"):
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
        
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        self.fh = logging.FileHandler(self.log_path)
        self.fh.setLevel(logging.DEBUG)

        # Console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)

        # Formatter
        #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter = ColorFormatter()
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)

        # Add handlers (prevent duplicate handlers)
        if not self.logger.handlers:
            self.logger.addHandler(self.fh)
            self.logger.addHandler(self.ch)

    def set_level(self, level):
        """Change the logging level dynamically."""
        self.logger.setLevel(level)
        self.fh.setLevel(level)
        self.ch.setLevel(level)

    def get_logger(self):
        """Return the logger instance."""
        return self.logger


class ColorFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s " #(%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
