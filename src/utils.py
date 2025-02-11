import mlflow
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.utilities import rank_zero_only
from copy import deepcopy
from typing import List

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


class FilteringMlFlowLogger(MLFlowLogger):
    def __init__(self, filter_submodels: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.filter_submodels = filter_submodels

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        params = deepcopy(params)
        [params.model.pop(filter_submodel) for filter_submodel in self.filter_submodels if filter_submodel in params.model]
        super().log_hyperparams(params)


def create_loaders_with_indices(full_dataset, train_val_split, batch_size, num_workers, seed):
    """
    Splits the dataset into training and validation subsets, creates data loaders, and returns the indices of each subset.

    Args:
        full_dataset (Dataset): The complete dataset to split.
        train_val_split (float): Proportion of the dataset to use for training (e.g., 0.8 for 80% training).
        batch_size (int): Batch size for both training and validation loaders.
        num_workers (int): Number of subprocesses to use for data loading.
        seed (int): Seed for reproducibility of the split.

    Returns:
        tuple: Contains training DataLoader, validation DataLoader, training indices, validation indices.
    """
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * (1 - train_val_split))
    train_size = dataset_size - val_size

    # Split the dataset into training and validation subsets
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Extract the indices from the subsets
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, train_indices, val_indices