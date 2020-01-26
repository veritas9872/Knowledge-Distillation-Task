from pathlib import Path
from datetime import datetime
import argparse
import torch


def initialize(record_root: str, train_method: str):
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


def get_arg_parser(**args):
    parser = argparse.ArgumentParser(description='Simple argument parser for placing default arguments as desired.')
    parser.set_defaults(**args)
    return parser


def get_gpu_if_available(gpu: int = None):
    # Device agnostic setting.
    return torch.device(f'cuda:{gpu}') if torch.cuda.is_available() and (gpu is not None) else torch.device('cpu')