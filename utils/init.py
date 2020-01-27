from datetime import datetime
from pathlib import Path


def initialize(record_root: str, train_method: str):
    """
    Initialize the folders for storing logs and checkpoints. Also decides the names of runs.
    Args:
        record_root: root location for storing records.
        train_method: method used for training.

    Returns:

    """
    record_root = Path(record_root)
    record_root.mkdir(exist_ok=True)

    record_path = record_root / train_method
    record_path.mkdir(exist_ok=True)

    run_number = sum(location.is_dir() for location in record_path.iterdir()) + 1
    time_string = str(datetime.now()).split('.')[0].replace(':', '-')
    run_name = f'Trial {run_number:02d}  {time_string}'

    record_path = record_path / run_name
    record_path.mkdir(exist_ok=False)

    checkpoint_path = record_path / 'checkpoints'
    log_path = record_path / 'logs'

    checkpoint_path.mkdir(exist_ok=False)
    log_path.mkdir(exist_ok=False)

    print(f'Created directories for {record_path}')
    print('Starting', run_name)

    return run_number, run_name, log_path, checkpoint_path


