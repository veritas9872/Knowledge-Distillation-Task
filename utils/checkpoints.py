from pathlib import Path

import torch
from torch import nn, optim


class CheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    Do not confuse with the Pytorch checkpoint module, which is not about saving the model for later use.
    Note that the whole system is based on 1 based indexing, not 0 based indexing.
    """
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, checkpoint_path: str,
                 mode='min', save_best_only=False, max_to_keep=5, verbose=True):

        # Input checking.
        assert isinstance(max_to_keep, int) and (max_to_keep >= 0), 'Not a non-negative integer'
        assert mode in ('min', 'max'), 'Mode must be either "min" or "max"'
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise OSError('Not a valid, existing path')

        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.save_best_only = save_best_only
        self.max_to_keep = max_to_keep
        self.verbose = verbose
        self.save_counter = 0
        self.record_dict = dict()

        if mode == 'min':
            self.prev_best = float('inf')
        elif mode == 'max':
            self.prev_best = -float('inf')
        else:
            raise TypeError('Mode must be either "min" or "max"')

    def _save(self, name=None, **save_kwargs):
        self.save_counter += 1
        save_dict = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        save_dict.update(save_kwargs)
        save_path = self.checkpoint_path / (f'{name}.tar' if name else f'checkpoint_{self.save_counter:03d}.tar')

        torch.save(save_dict, save_path)
        if self.verbose:
            print(f'Saved Checkpoint to {save_path}')
            print(f'Checkpoint {self.save_counter:04d}: {save_path}')

        self.record_dict[self.save_counter] = save_path

        if self.save_counter > self.max_to_keep:
            for count, checkpoint_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and checkpoint_path.exists():
                    checkpoint_path.unlink()  # Delete existing checkpoint

        return save_path

    def save(self, metric, name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either "min" or "max"')

        save_path = None
        if (self.max_to_keep > 0) and (is_best or not self.save_best_only):
            save_path = self._save(name, **save_kwargs)

        if self.verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_path, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dir, strict=True, load_optimizer=True):  # Load to continue training.
        save_dict = torch.load(load_dir)

        self.model.load_state_dict(save_dict['model_state_dict'], strict=strict)
        print(f'Loaded model parameters from {load_dir}, strict={strict}')

        if load_optimizer:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'], strict=strict)
            print(f'Loaded optimizer parameters from {load_dir}, strict={strict}')

    def load_latest(self, load_dir, strict=True, load_optimizer=True):
        load_dir = Path(load_dir)
        checkpoints = [file for file in sorted(load_dir.iterdir()) if file.is_dir()]

        if not checkpoints:
            raise FileNotFoundError('The given directory has no valid files.')

        load_file = checkpoints[-1]

        print('Loading', load_file)
        self.load(load_file, strict=strict, load_optimizer=load_optimizer)
        print('Done')


def load_model_from_checkpoint(model: nn.Module, load_dir: str):
    """
    A simple function for loading checkpoints without having to use Checkpoint Manager.
    Very useful for evaluation since checkpoint manager was designed for loading checkpoints before resuming training.
    Args:
        model: model architecture whose values are to be restored from the checkpoint.
        load_dir: the file path to the checkpoint
    """
    assert Path(load_dir).exists(), 'The specified directory does not exist'
    save_dict = torch.load(load_dir, map_location=model.device)  # Allow dynamic reloading to any device.
    model.load_state_dict(save_dict['model_state_dict'])
