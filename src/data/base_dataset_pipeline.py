import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from sklearn.model_selection import KFold, train_test_split
import logging
logger = logging.getLogger(__name__)

class ProcessedDataset(Dataset):
    """
    Dataset object for the processed data fed to the training of the model
    """
    def __init__(self, Y, T_disc = None, T_cont = None, X_static = None, X_dynamic = None, active_entries = None, subset_name = 'train'):
        self.subset_name = subset_name
        self.Y = Y
        self.T_disc = T_disc
        self.T_cont = T_cont
        self.X_static = X_static
        self.X_dynamic = X_dynamic
        assert self.X_dynamic is not None
        #change type of active_entries to boolean
        self.active_entries = active_entries if active_entries is not None \
            else torch.ones_like(Y, dtype=torch.float32)
        
        self.prev_Y = torch.zeros_like(Y)
        self.prev_Y[:, 1:] = Y[:, :-1]
        self.prev_T_disc = torch.zeros_like(T_disc) if T_disc is not None else None
        self.prev_T_cont = torch.zeros_like(T_cont) if T_cont is not None else None

        self.disc_dim = T_disc.shape[2] if T_disc is not None else 0
        self.cont_dim = T_cont.shape[2] if T_cont is not None else 0
        self.static_dim = X_static.shape[1] if X_static is not None else 0

        self.res_Y = None
        self.res_T_disc = None
        self.res_T_cont = None

        self.gt_dynamic_effect_available = False #In some models, the true dynamic effect is available

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return {
            "curr_treatments_disc": self.T_disc[idx] if self.T_disc is not None else torch.zeros(0),      # (S, n_treatments_disc)
            "curr_treatments_cont": self.T_cont[idx] if self.T_cont is not None else torch.zeros(0),      # (S, n_treatments_cont)
            "prev_treatments_disc": self.prev_T_disc[idx] if self.prev_T_disc is not None else torch.zeros(0), # (S, n_treatments_disc)
            "prev_treatments_cont": self.prev_T_cont[idx] if self.prev_T_cont is not None else torch.zeros(0), # (S, n_treatments_cont)
            "curr_outputs": self.Y[idx],         # (S,)
            "prev_outputs": self.prev_Y[idx],    # (S,)
            'active_entries': self.active_entries[idx], # (S,)
            'static_features': self.X_static[idx] if self.X_static is not None else torch.zeros(0), # (S, n_static)
            "curr_covariates": self.X_dynamic[idx],       # (S, n_x)
            "residual_Y": self.res_Y[idx] if self.res_Y is not None else torch.zeros(0),
            "residual_T_disc": self.res_T[idx] if self.res_T is not None else torch.zeros(0),
            "residual_T_cont": self.res_T[idx] if self.res_T is not None else torch.zeros(0)
        }
    
    def add_residual_data(self, subset_index, res_Y, res_T_disc = None, res_T_cont = None):
        """
        Add residuals to the dataset by the subset index"""

        #sanity check, T_disc/T_cont and res_T_disc/res_T_cont should be/not be None at the same time
        if res_T_disc is not None and (res_T_disc is None):
            raise ValueError("Treatments residuals mis-match")
        if res_T_disc is None and (res_T_disc is not None):
            raise ValueError("Treatments residuals mis-match")
        if res_T_cont is not None and (res_T_cont is None):
            raise ValueError("Treatments residuals mis-match")
        if res_T_cont is None and (res_T_cont is not None):
            raise ValueError("Treatments residuals mis-match")

        if self.res_Y is None:
            self.res_Y = torch.zeros_like(self.Y)
        self.res_Y[subset_index] = res_Y

        if res_T_disc is not None:
            if self.res_T_disc is None:
                self.res_T_disc = torch.zeros_like(self.T_disc)
            self.res_T_disc[subset_index] = res_T_disc
        if res_T_cont is not None:
            if self.res_T_cont is None:
                self.res_T_cont = torch.zeros_like(self.T_cont)
            self.res_T_cont[subset_index] = res_T_cont
        return


class BaseDatasetPipeline:
    def __init__(self, argsdataset):
        """
        Initialize basic configurations and set parameters.
        Args:
            config (dict): Configuration dict/object containing global parameters
                           e.g. seed, train/val/test split ratios, etc.
        """
        self.param = argsdataset
        self.seed = argsdataset.get('seed', 42)
        self.n_treatments = argsdataset['n_treatments']
        self.n_treatments_disc = argsdataset.get('n_treatments_disc', 0)
        self.n_treatments_cont = argsdataset.get('n_treatments_cont', 0)
        assert self.n_treatments == self.n_treatments_disc + self.n_treatments_cont, "Mismatch in treatment count."
        self.n_units = argsdataset['n_units']
        self.n_periods = argsdataset['n_periods']
        self.sequence_length = argsdataset['sequence_length']
        self.val_split, self.test_split = argsdataset['split']['val'], argsdataset['split']['test']
        self.kfold_split = None
        self.fold_num = 0
        self.loaded = False #whether the data has been loaded
        self.factual_generated = False #whether the factual data has been generated

        self.factual_data = None  # Observational data; type depends on dataset (numpy or pandas)
        self.train_index, self.val_index, self.test_index = None, None, None
        self.index = {'train': None, 'val': None, 'test': None}
        self.train_data, self.val_data, self.test_data = None, None, None #dataset objects
        self.k_fold_split = None
        self.counterfactual_data = None  # Will be set once simulated
        self.true_effect = None

    def load_data(self):
        """
        Load raw data.
        For semi-synthetic datasets, this can include reading a DataFrame from disk.
        For fully synthetic datasets, this function might generate all the data.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _generate_full_factual_data(self, **kargs):
        """
        Generate the observational (factual) full data (train + val + test).
        For fully synthetic, simulate all outcomes.
        For semi-synthetic, perform the necessary processing.
        The internal representation could be a numpy array or a pandas DataFrame.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _split_data(self, **kargs):
        """
        Split the factual data into train, validation, and test subsets.
        This function should maintain the indices for reproducibility.
        Args:
            train_val_test_ratio (tuple): e.g. (0.7, 0.15, 0.15)
        Returns:
            splits (dict): Dictionary with keys 'train', 'val', 'test'.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_torch_dataset(self, subset='train') -> ProcessedDataset:
        """
        Wrap up the specified data subset as a torch.utils.data.Dataset.
        Args:
            subset (str): One of 'train', 'val', 'test'
        Returns:
            torch_dataset (Dataset): A torch Dataset object containing the specified subset.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _simulate_counterfactuals(self, treatment_seq, subset_name):
        """
        Simulate counterfactual data given a fixed treatment sequence.
        The simulation is performed on the specified subset (train, val or test).
        The result is stored internally.
        Args:
            treatment_seq (np.array): The intervention treatment sequence(s).
            subset_name (str): Subset identifier, e.g., 'train', 'val', or 'test'.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_treatment_effect(self, subset, intv_T_disc, intv_T_cont, base_T_disc, base_T_cont):
        """
        Args:
            subset: str, 'train', 'val', 'test'
            intervention_T: np.ndarray of shape (n_periods,)
            baseline_T: np.ndarray of shape (n_periods,)
            
        Returns:
            np.ndarray: Treatment effect matrix of shape (n_units, sequence_length - n_periods + 1)
        """
        intervention_T = self._combine_disc_cont(intv_T_disc, intv_T_cont)
        baseline_T = self._combine_disc_cont(base_T_disc, base_T_cont)
        assert intervention_T.shape == (self.n_periods, self.n_treatments)
        assert baseline_T.shape == (self.n_periods, self.n_treatments)
        
        # Simulate counterfactuals
        Y_intervention = self._simulate_counterfactuals(subset, intervention_T)
        Y_baseline = self._simulate_counterfactuals(subset, baseline_T)
        
        # Calculate treatment effects starting from t=n_periods-1
        return Y_intervention - Y_baseline
    
    def compute_individual_true_dynamic_effects(self) -> np.array:
        """
        Compute the true individual dynamic treatment effect.
        Returns:
            true_effect (np.array or pd.DataFrame): True individual treatment effects.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def shuffle_data(self):
        """
        Shuffle the factual dataset (e.g., randomize the order of the units).
        The structure of indices should be maintained.
        Must be implemented by subclasses based on the internal representation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def k_fold_split(self, k=3):
        """
        Perform a k-fold cross-validation split on the training data.
        Yields indices for local train and validation subsets on each split.
        Args:
            k (int): Number of folds.
        Yields:
            train_indices, val_indices (tuple of np.ndarray): Indices for each fold.
        """
        if self.factual_generated == False:
            raise ValueError("Factual data not generated. Run generate_factual_data() first.")
        if self.train_data is None:
            raise ValueError("Training data not yet available. Run split_data() first.")
        
        #perform k-fold split on the training data
        total_samples = self._get_train_size()
        self.fold_num = k
        
        indices = np.arange(total_samples)
        kf = KFold(n_splits=k, shuffle=False, random_state=self.seed)
        self.k_fold_split = {split_idx:None for split_idx in range(k)}
        for split_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            self.k_fold_split[split_idx] = (train_idx, val_idx)
            yield Subset(self.train_data, train_idx), Subset(self.train_data, val_idx)
    
    def _get_train_size(self):
        if self.train_index is not None:
            return len(self.train_index)
        else:
            raise ValueError("Training data not yet split. Run split_data() first.")

    def add_fold_residual_data(self, fold_idx, res_Y, res_T_disc = None, res_T_cont = None):
        """
        Add residuals (from first-stage estimation) to the training set of the pipeline.
        This should combine or update the existing factual data.
        Args:
            res_Y (np.array or pd.Series): Residuals for outputs from one fold.
            res_T_cont / res_T_disc (np.array or pd.Series): Residuals for treatments from one fold.
        """
        indices = self.k_fold_split[fold_idx][1]
        if self.train_data is not None:
            self.train_data.add_residual_data(indices, res_Y, res_T_disc, res_T_cont)
        else:
            raise ValueError("Training data not yet available. ")
        logger.info(f"Residual data of fold {fold_idx} added to training set.")

        return
    
    def add_full_residual_data(self, subset, res_Y, res_T_disc = None, res_T_cont = None):
        """
        Add residuals (from first-stage estimation) to the full set (train/val/test) of the pipeline.
        This should combine or update the existing factual data.
        Args:
            res_Y (np.array or pd.Series): Residuals for outputs.
            res_T_cont / res_T_disc (np.array or pd.Series): Residuals for treatments.
        """
        if subset == 'train':
            self.train_data.add_residual_data(np.arange(len(self.train_data)), res_Y, res_T_disc, res_T_cont)
        elif subset == 'val':
            self.val_data.add_residual_data(np.arange(len(self.val_data)), res_Y, res_T_disc, res_T_cont)
        elif subset == 'test':
            self.test_data.add_residual_data(np.arange(len(self.test_data)), res_Y, res_T_disc, res_T_cont)
        else:
            raise ValueError("Invalid subset name. Choose from 'train', 'val', 'test'.")
            
        return
    
    def _combine_disc_cont(self, T_disc:np.array, T_cont:np.array):
        """
        Combine discrete and continuous treatments
        """
        if T_disc is None:
            return T_cont
        elif T_cont is None:
            return T_disc
        else:
            assert T_disc.shape[:-1] == T_cont.shape[:-1]
            return np.concatenate([T_disc, T_cont], dim = -1)
    
    