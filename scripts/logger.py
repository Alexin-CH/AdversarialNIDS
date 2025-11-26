import logging
import os
from datetime import datetime

class SimpleLogger:
    def debug(self, message):
        """Log a debug message."""
        print(f"[DEBUG] {message}")
    
    def info(self, message):
        """Log an info message."""
        print(f"[INFO] {message}")
    
    def warning(self, message):
        """Log a warning message."""
        print(f"[WARNING] {message}")
    
    def error(self, message):
        """Log an error message."""
        print(f"[ERROR] {message}")
    
    def critical(self, message):
        """Log a critical message."""
        print(f"[CRITICAL] {message}")

class LoggerManager:
    """Logger Manager to handle logging to file and console with timestamps."""
    def __init__(self, root_dir='.', log_name="log"):
        """
        Initialize the LoggerManager.

        Args:
            root_dir (str): Root directory to save logs.
            log_name (str): Base name for the log file.
        """
        log_dir = os.path.join(root_dir, "results", "logs")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        self.title = f"{log_name}_{timestamp}"
        self.log_path = os.path.join(log_dir, f"{self.title}.log")
        
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        self.fh = logging.FileHandler(self.log_path)
        self.fh.setLevel(logging.DEBUG)

        # Console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)

        # Add handlers (prevent duplicate handlers)
        if not self.logger.handlers:
            self.logger.addHandler(self.fh)
            self.logger.addHandler(self.ch)

    def set_level(self, level=logging.INFO):
        """Change the logging level dynamically."""
        self.logger.setLevel(level)
        self.fh.setLevel(level)
        self.ch.setLevel(level)

    def get_logger(self):
        """Return the logger instance."""
        return self.logger

    def get_title(self):
        """Return the log title."""
        return self.title
    