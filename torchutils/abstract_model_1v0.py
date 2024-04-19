"""
Abstract model to ease device selection, determinism, and checkpoint managment.

This abstract model must be subclassed and redefine the following attributes:
    - project_path
    - network
And optionally:
    - seed
    - dtype
    - maximum_saves
    - optimizer
    - scheduler

Author: Pierre Lelievre
"""

import os
import torch
import numpy as np


# Abstract model


class AbstractModel:
    # Default parameters : to be defined ######################################
    project_path = None
    # Default parameters : optional ###########################################
    seed = None
    dtype = torch.float32
    maximum_saves = 1
    ###########################################################################
    def __init__(self, model_name, device=None, parameters=None,
                 model_name_suffix='', enforce_determinisim=True):
        # Enforce determinism
        if enforce_determinisim:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if enforce_determinisim and (self.seed is None):
            self.seed = 100
        # Update parameters
        if parameters is not None:
            for key, value in parameters.items():
                setattr(self, key, value)
        # Initt logs, saves, results folders
        self.model_name = model_name
        self.model_name_suffix = model_name_suffix
        self.logs_path = None
        self.saves_path = None
        self.results_path = None
        self._init_paths()
        # Set state variables
        self.current_step = 0
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_val_acc = 1e10
        # Set Seeds
        self.rng = None
        self.set_seed(self.seed)
        # Set PyTorch default dtype
        torch.set_default_dtype(self.dtype)
        # Set device
        self.device = torch.device('cpu')
        if isinstance(device, int):
            device = f'cuda:{device}'
        if isinstance(device, str):
            if len(device) >= 4 and device[:4] == 'cuda' and (
                    torch.cuda.is_available()):
                self.device = torch.device(device)
                torch.cuda.device(self.device)
            elif device == 'mps' and torch.backends.mps.is_available():
                self.device = torch.device(device)
            elif device == 'cpu':
                self.device = torch.device(device)
            else:
                raise ValueError(f'Device {device} is not available.')
        # Network : to be defined #############################################
        self.network = None
        # Optimizer & scheduler : optional ####################################
        self.optimizer, self.scheduler = None, None
        #######################################################################

    def _init_paths(self):
        logs_path = os.path.join(self.project_path, 'logs')
        if not os.path.isdir(logs_path):
            os.mkdir(logs_path)
        saves_path = os.path.join(self.project_path, 'saves')
        if not os.path.isdir(saves_path):
            os.mkdir(saves_path)
        results_path = os.path.join(self.project_path, 'results')
        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        self.logs_path = os.path.join(
            logs_path, self.model_name + self.model_name_suffix)
        if not os.path.isdir(self.logs_path):
            os.mkdir(self.logs_path)
        self.saves_path = os.path.join(
            saves_path, self.model_name + self.model_name_suffix)
        if not os.path.isdir(self.saves_path):
            os.mkdir(self.saves_path)
        self.results_path = os.path.join(
            results_path, self.model_name + self.model_name_suffix)
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)

    def set_seed(self, seed=None):
        """
        Set seed of the default PyTorch random number generator (used for
        weights initialization) and self.rng (a numpy random number generator).
        """
        if seed is not None:
            torch.manual_seed(seed)
            self.rng = np.random.default_rng(seed)

    def save(self, val_acc):
        """
        Save current weights of the network and update state variables.
        Save optimizer and scheduler parameters, if defined.
        """
        if val_acc < self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = self.current_epoch
        # Checkpoint data
        checkpoint_path = os.path.join(
            self.saves_path, f'checkpoint-{self.current_epoch:04d}')
        state = {
            'epoch': self.current_epoch,
            'step' : self.current_step,
            'val_acc': val_acc,
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'state_dict': self.network.state_dict()}
        torch.save(state, checkpoint_path)
        # Training checkpoint data
        if self.optimizer is None:
            return
        checkpoint_training_path = os.path.join(
            self.saves_path, f'checkpoint_training-{self.current_epoch:04d}')
        state_training = {'optimizer': self.optimizer.state_dict()}
        if self.scheduler is not None:
            state_training.update({'scheduler': self.scheduler.state_dict()})
        torch.save(state_training, checkpoint_training_path)

    def clean_saves(self):
        """
        Clean up saves folder with self.maximum_saves best checkpoints and the
        last checkpoint (required to continue training).
        """
        checkpoint_val_acc = []
        for f in sorted(os.listdir(self.saves_path)):
            if not os.path.isfile(os.path.join(self.saves_path, f)):
                continue
            f_split = f.split('-')
            if f_split[0] != 'checkpoint':
                continue
            checkpoint = int(f_split[-1])
            checkpoint_data = torch.load(os.path.join(
                self.saves_path, f'checkpoint-{checkpoint:04d}'),
                map_location=torch.device('cpu'))
            checkpoint_val_acc.append((checkpoint, checkpoint_data['val_acc']))
        if len(checkpoint_val_acc) - 1 > self.maximum_saves:
            checkpoints = [i[0] for i in sorted(
                checkpoint_val_acc[:-1], key=lambda val_acc: val_acc[1])]
            for checkpoint in checkpoints[self.maximum_saves:]:
                checkpoint_path = os.path.join(
                    self.saves_path, f'checkpoint-{checkpoint:04d}')
                if os.path.isfile(checkpoint_path):
                    os.remove(checkpoint_path)
                checkpoint_training_path = os.path.join(
                    self.saves_path, f'checkpoint_training-{checkpoint:04d}')
                if os.path.isfile(checkpoint_training_path):
                    os.remove(checkpoint_training_path)

    def _get_checkpoint_data(self, checkpoint=None, training=False):
        if checkpoint is None:
            return None
        assert checkpoint in ('last', 'best') or isinstance(checkpoint, int), (
            "Checkpoint must be 'last', 'best', or an integer.")
        checkpoints = []
        for f in sorted(os.listdir(self.saves_path)):
            if os.path.isfile(os.path.join(self.saves_path, f)):
                f_split = f.split('-')
                if f_split[0] == 'checkpoint':
                    checkpoints.append(int(f_split[-1]))
        if training and not checkpoints:
            return None
        if not checkpoints:
            raise ValueError('No checkpoint to restore.')
        if checkpoint == 'last':
            checkpoint = checkpoints[-1]
        elif checkpoint == 'best':
            checkpoint_data = torch.load(os.path.join(
                    self.saves_path, f'checkpoint-{checkpoints[-1]:04d}'),
                map_location=torch.device('cpu'))
            checkpoint = checkpoint_data['best_epoch']
        if checkpoint != self.current_epoch:
            if checkpoint not in checkpoints:
                raise ValueError(
                    f'Checkpoint {checkpoint:04d} does not exist.')
            checkpoint_data = torch.load(
                os.path.join(self.saves_path, f'checkpoint-{checkpoint:04d}'),
                map_location=self.device)
            return checkpoint_data
        return None

    def restore(self, checkpoint=None, training=False):
        """
        Restore a particular checkpoint. Checkpoint can be:
            - best checkpoint ('best')
            - last checkpoint ('last')
            - a specific checkpoint (int)
        If training is True, optimizer and scheduler parameters are restored.
        """
        checkpoint_data = self._get_checkpoint_data(checkpoint, training)
        if checkpoint_data is None:
            return
        self.current_epoch = checkpoint_data['epoch']
        self.current_step = checkpoint_data['step']
        self.best_epoch = checkpoint_data['best_epoch']
        self.best_val_acc = checkpoint_data['best_val_acc']
        self.network.load_state_dict(checkpoint_data['state_dict'])
        if self.optimizer is not None:
            checkpoint = checkpoint_data['epoch']
            checkpoint_training_path = os.path.join(
                self.saves_path, f'checkpoint_training-{checkpoint:04d}')
            checkpoint_training_data = torch.load(
                os.path.join(self.saves_path, checkpoint_training_path),
                map_location=self.device)
            self.optimizer.load_state_dict(
                checkpoint_training_data['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(
                checkpoint_training_data['scheduler'])
        print(f'Restored checkpoint: {self.current_epoch} '
              f'(best: {self.best_epoch})')

    def get_result_path(self, name):
        """
        Return the path of an item 'name' located in results folder.
        """
        return os.path.join(self.results_path, name)

    def get_new_result_path(self, prefix, extension, separator='-'):
        """
        Return the path of an item 'prefix.extension' located in results
        folder. If this item already exists, it will be incremented, such as
        'prefix|separator|000.extension'.
        """
        i = 0
        path = os.path.join(
            self.results_path, f'{prefix}{separator}{i:03d}.{extension}')
        while os.path.isfile(path):
            i += 1
            path = os.path.join(
                self.results_path, f'{prefix}{separator}{i:03d}.{extension}')
        return path

    def clean_empty_folders(self, extra_folder_names=None):
        """
        Clean up empty model folders located in logs, saves, and results
        folders, that are created by defaults at the initalization of the
        model.
        """
        folder_names = ('logs', 'saves', 'results')
        if extra_folder_names is not None:
            folder_names += extra_folder_names
        for f_0 in os.listdir(self.project_path):
            if f_0 in folder_names:
                p_0 = os.path.join(self.project_path, f_0)
                for f_1 in os.listdir(p_0):
                    p_1 = os.path.join(p_0, f_1)
                    if os.path.isdir(p_1) and not os.listdir(p_1):
                        os.rmdir(p_1)
        return self
