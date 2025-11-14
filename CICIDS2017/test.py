import os
import sys
from datetime import datetime

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.logger import LoggerManager
from CICIDS2017.preprocessing import preprocess_cicids2017, data_encoding, optimize_memory_usage

date = datetime.now().strftime("%y%m%d_%H%M%S")
title = "test"
lm = LoggerManager(log_dir=f"{current_dir}/logs", log_name=title)
lm.logger.info("Logger initialized")

data = preprocess_cicids2017(logger=lm.logger)
data = data_encoding(data, logger=lm.logger)
data = optimize_memory_usage(data, logger=lm.logger)