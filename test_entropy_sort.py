from scripts.logger import LoggerManager
from scripts.feature_selection.entropy_sort import (
	_load_dataframe_from_preprocessing,
	rank_features,
)
import pandas as pd


def main():
	# Create logger for the test run
	mgr = LoggerManager(log_dir="logs", log_name="test_entropy_sort")
	logger = mgr.get_logger()

	# Load and preprocess dataset (preprocess_cicids2017 is called internally)
	df = _load_dataframe_from_preprocessing(logger=logger)

	# Compute feature ranking
	ranking = rank_features(df, logger=logger)

	# Display and log top features
	pd.set_option('display.max_rows', 50)
	top = ranking.head(10)
	logger.info("Top 10 features:\n%s", top.to_string(index=False))


if __name__ == '__main__':
	main()