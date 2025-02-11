import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold, train_test_split


class BaseDatasetPipeline:
    def __init__(self, config):
        """
        Initialize basic configurations and set parameters.
        Args:
            config (dict): Configuration dict/object containing global parameters
                           e.g. seed, train/val/test split ratios, etc.
        """
        self.config = config
        self.seed = config.get('seed', 42)
        np.random.seed(self.seed)
        self.factual_data = None  # Observational data; type depends on dataset (numpy or pandas)
        self.counterfactual_data = None  # Will be set once simulated
        self.residual_data = None  # Tuple (res_Y, res_T) if available

    def load_data(self):
        """
        Load raw data.
        For semi-synthetic datasets, this can include reading a DataFrame from disk.
        For fully synthetic datasets, this function might generate all the data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_factual_data(self, **kargs):
        """
        Generate the observational (factual) data.
        For fully synthetic, simulate all outcomes.
        For semi-synthetic, perform any necessary processing.
        The internal representation could be a numpy array or a pandas DataFrame.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def split_data(self, train_val_test_ratio):
        """
        Split the factual data into train, validation, and test subsets.
        This function should maintain the indices for reproducibility.
        Args:
            train_val_test_ratio (tuple): e.g. (0.7, 0.15, 0.15)
        Returns:
            splits (dict): Dictionary with keys 'train', 'val', 'test'.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_torch_dataset(self, subset='train'):
        """
        Wrap up the specified data subset as a torch.utils.data.Dataset.
        Args:
            subset (str): One of 'train', 'val', 'test'
        Returns:
            torch_dataset (Dataset): A torch Dataset object containing the specified subset.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def simulate_counterfactuals(self, treatment_seq, subset_name):
        """
        Simulate counterfactual data given a fixed treatment sequence.
        The simulation is performed on the specified subset (train, val or test).
        The result is stored internally.
        Args:
            treatment_seq (np.array): The intervention treatment sequence(s).
            subset_name (str): Subset identifier, e.g., 'train', 'val', or 'test'.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_treatment_effect(self, treatment_intv, treatment_base):
        """
        Compute the treatment effect using the counterfactual trajectories.
        Args:
            treatment_intv: Intervention treatment definition.
            treatment_base: Baseline treatment definition.
        Returns:
            treatment_effect (np.array or pd.DataFrame): Estimated treatment effects.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def shuffle_data(self):
        """
        Shuffle the factual dataset (e.g., randomize the order of the units).
        The structure of indices should be maintained.
        Must be implemented by subclasses based on the internal representation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def k_fold_split(self, k=5):
        """
        Perform a k-fold cross-validation split on the training data.
        Yields indices for local train and validation subsets on each split.
        Args:
            k (int): Number of folds.
        Yields:
            train_indices, val_indices (tuple of np.ndarray): Indices for each fold.
        """
        if self.factual_data is None:
            raise ValueError("Factual data not generated. Run generate_factual_data() first.")
        
        # Assuming self.factual_data can be indexed as a numpy array or supports .index for pandas
        if isinstance(self.factual_data, pd.DataFrame):
            total_samples = len(self.factual_data)
        else:
            total_samples = self.factual_data.shape[0]
        
        indices = np.arange(total_samples)
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        for train_idx, val_idx in kf.split(indices):
            yield train_idx, val_idx

    def add_residual_data(self, res_Y, res_T):
        """
        Add residuals (from first-stage estimation) to the pipeline.
        This should combine or update the existing factual data.
        Args:
            res_Y (np.array or pd.Series): Residuals for outputs.
            res_T (np.array or pd.Series): Residuals for treatments.
        """
        self.residual_data = (res_Y, res_T)
        # Implementation detail: update factual_data with residuals.
        raise NotImplementedError("Subclasses should implement how residuals are added to the dataset.")