import pandas as pd
from pathlib import Path
from .utils import get_filename_no_ext

# exports
__all__ = ['competition_name', 'target_cols', 'feature_cols', 'input_dir', 'output_dir',
           'sample_submission', 'tdcsfog_metadata', 'defog_metadata', 'daily_metadata', 'subjects', 'events', 'tasks',
           'tdcsfog_train_dir', 'defog_train_dir', 'notype_train_dir', 'tdcsfog_test_dir', 'defog_test_dir',
           'tdcsfog_train_sessions', 'defog_train_sessions', 'notype_train_sessions', 'tdcsfog_test_sessions', 'defog_test_sessions']


# IO
competition_name = "tlvmc-parkinsons-freezing-gait-prediction"
target_cols = ["StartHesitation", "Turn", "Walking"]
feature_cols = ["AccV", "AccML", "AccAP"]
input_dir = Path("../input/") / competition_name
output_dir = Path("./")

# Dataframes
sample_submission = pd.read_csv(input_dir / "sample_submission.csv")
tdcsfog_metadata = pd.read_csv(input_dir / "tdcsfog_metadata.csv")
defog_metadata = pd.read_csv(input_dir / "defog_metadata.csv")
daily_metadata = pd.read_csv(input_dir / "daily_metadata.csv")
subjects = pd.read_csv(input_dir / "subjects.csv")
events = pd.read_csv(input_dir / "events.csv")
tasks = pd.read_csv(input_dir / "tasks.csv")

# Directories
tdcsfog_train_dir = input_dir / 'train/tdcsfog'
defog_train_dir = input_dir / 'train/defog'
notype_train_dir = input_dir / 'train/notype'
tdcsfog_test_dir = input_dir / 'test/tdcsfog'
defog_test_dir = input_dir / 'test/defog'


# Sessions for each dataset
tdcsfog_train_sessions = [get_filename_no_ext(p) for p in tdcsfog_train_dir.glob('*.csv')]
defog_train_sessions = [get_filename_no_ext(p) for p in defog_train_dir.glob('*.csv')]
notype_train_sessions = [get_filename_no_ext(p) for p in notype_train_dir.glob('*.csv')]
tdcsfog_test_sessions = [get_filename_no_ext(p) for p in tdcsfog_test_dir.glob('*.csv')]
defog_test_sessions = [get_filename_no_ext(p) for p in defog_test_dir.glob('*.csv')]