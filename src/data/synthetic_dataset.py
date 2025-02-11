import numpy as np
from econml.utilities import cross_product
from statsmodels.tools.tools import add_constant
import torch
from torch.utils.data import Dataset, random_split
import logging
logger = logging.getLogger(__name__)

class ProcessedDataset(Dataset):
    def __init__(self, Y, T, X, prev_Y, prev_T):
        self.Y = Y
        self.T = T
        self.X = X
        self.prev_Y = prev_Y
        self.prev_T = prev_T
        self.res_Y = None
        self.res_T = None

    def add_residual_data(self, res_Y, res_T):
        assert self.Y.shape[0] == res_Y.shape[0]
        assert self.Y.shape[0] == res_T.shape[0]
        self.res_Y = res_Y
        self.res_T = res_T

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return {
            "curr_treatments": self.T[idx],      # (S, n_treatments)
            "prev_treatments": self.prev_T[idx], # (S, n_treatments)
            "curr_outputs": self.Y[idx],         # (S,)
            "prev_outputs": self.prev_Y[idx],    # (S,)
            "curr_covariates": self.X[idx],       # (S, n_x)
            "residual_Y": self.res_Y[idx] if self.res_Y is not None else torch.zeros(0),
            "residual_T": self.res_T[idx] if self.res_T is not None else torch.zeros(0)
        }

class MarkovianHeteroDynamicDataset():
    """
    Uses code from Dynamic DML (DynamicPancelDGP class) 
    https://proceedings.neurips.cc/paper/2021/hash/bf65417dcecc7f2b0006e1f5793b7143-Abstract.html
    """

    def __init__(self, params):
        np.random.seed(params['seed'])
        self.n_treatments = params['n_treatments']
        self.n_units = params['n_units']
        self.n_periods = params['n_periods']
        self.sequence_length = params['sequence_length']
        self.sigma_x = params['sigma_x']
        self.sigma_y = params['sigma_y']
        self.n_x = params['n_x']
        self.state_effect = params['state_effect']
        self.autoreg = params['autoreg']
        self.hetero_strength = params['hetero_strength']
        self.conf_str = params['conf_str']
        self.s_x = params['s_x']
        self.train_val_split = params['train_val_split']
        self.params = params

        self.hetero_inds = np.array(params.hetero_inds, dtype=np.int32) if (
                                        (params.hetero_inds is not None) and (len(params.hetero_inds) > 0)) else None
        self.endo_inds = np.setdiff1d(np.arange(self.n_x), self.hetero_inds).astype(int)

        self.Alpha = np.random.uniform(-1, 1, size = (self.n_x, self.n_treatments))
        self.Alpha *= self.state_effect
        if self.hetero_inds is not None:
            self.Alpha[self.hetero_inds] = 0.
        
        self.Beta = np.zeros((self.n_x, self.n_x))
        for t in range(self.n_x):
            self.Beta[t, :] = self.autoreg * np.roll(np.random.uniform(low=4.0**(-np.arange(
                0, self.n_x)), high=4.0**(-np.arange(1, self.n_x + 1))), t)
        if self.hetero_inds is not None:
            self.Beta[np.ix_(self.endo_inds, self.hetero_inds)] = 0.
            self.Beta[np.ix_(self.hetero_inds, self.endo_inds)] = 0.
        
        self.epsilon = np.random.uniform(-1, 1, size = self.n_treatments)
        self.zeta = np.zeros(self.n_x)
        self.zeta[:self.s_x] = self.conf_str / self.s_x
        
        self.y_hetero_effect = np.zeros(self.n_x)
        self.x_hetero_effect = np.zeros(self.n_x)
        if self.hetero_inds is not None:
            self.y_hetero_effect[self.hetero_inds] = np.random.uniform(0.5 * self.hetero_strength, 1.5* self.hetero_strength) / len(self.hetero_inds)
            self.x_hetero_effect[self.hetero_inds] = np.random.uniform(0.5 * self.hetero_strength, 1.5* self.hetero_strength) / len(self.hetero_inds)

        self.true_effect = np.zeros((self.n_periods, self.n_treatments))
        self.true_effect[0] = self.epsilon
        for t in range(1, self.n_periods):
            self.true_effect[t, :] = self.zeta.reshape(1, -1) @ np.linalg.matrix_power(self.Beta, t - 1) @ self.Alpha
        
        self.true_hetero_effect = np.zeros((self.n_periods, (self.n_x + 1) * self.n_treatments))
        self.true_hetero_effect[0, :] = cross_product(add_constant(self.y_hetero_effect.reshape(1, -1), has_constant = 'add'), self.epsilon.reshape(1, -1))
        for t in np.arange(1, self.n_periods):
            self.true_hetero_effect[t, :] = cross_product(add_constant(self.x_hetero_effect.reshape(1, -1), has_constant='add'), 
                                                            self.zeta.reshape(1, -1) @ np.linalg.matrix_power(self.Beta, t - 1) @ self.Alpha)
        
        #store the random noises
        self.noisex = np.zeros((self.n_units, self.sequence_length, self.n_x))
        self.noisey = np.zeros((self.n_units, self.sequence_length))
        return

    def generate_observational_data(self, policy = None, seed = 2024):

        s_t = self.params["s_t"]
        sigma_t = self.params['sigma_t']
        gamma = self.params['gamma']

        self.Delta = np.zeros((self.n_treatments, self.n_x))
        self.Delta[:, :s_t] = self.conf_str / s_t

        if policy is None:
            def policy(Tprev, X, period):
                return gamma * Tprev + (1 - gamma) * self.Delta @ X + np.random.normal(0, sigma_t, size = self.n_treatments)
        
        np.random.seed(seed)
        Y = np.zeros((self.n_units, self.sequence_length))
        T = np.zeros((self.n_units, self.sequence_length, self.n_treatments))
        X = np.zeros((self.n_units, self.sequence_length, self.n_x))
        for i in range(self.n_units):
            for t in range(self.sequence_length):
                #Generate random exogeneous noise for X, y
                self.noisex[i, t] = np.random.normal(0, self.sigma_x, size = self.n_x)
                self.noisey[i, t] = np.random.normal(0, self.sigma_y)
                if t == 0:
                    X[i][0] = self.noisex[i, 0]
                    T[i][0] = policy(np.zeros(self.n_treatments), X[i][0], 0)
                else:
                    X[i][t] = (1 + np.dot(self.x_hetero_effect, X[i][t - 1])) * np.dot(self.Alpha, T[i][t - 1]) + \
                                np.dot(self.Beta, X[i][t - 1]) + self.noisex[i, t]
                    T[i][t] = policy(T[i][t - 1], X[i][t - 1], t)
                #Generate outcome
                Y[i][t] = (np.dot(self.y_hetero_effect, X[i][t]) + 1) * np.dot(self.epsilon, T[i][t]) + \
                            np.dot(X[i][t], self.zeta) + self.noisey[i, t]
        self.Y_obs = np.array(Y)
        self.T_obs = np.array(T)
        self.X_obs = np.array(X)
        return Y, T, X
    
    def compute_individual_dynamic_effects(self, X):
        """For linear markovian hetero datasets, we could compute the individual dynamic effects directly from self.true_hetero_effect
        Args:
            X: Covariate, np.ndarray of shape (N, SL, n_x) (the generated covariate)
        
        Returns:
            individual_de (N, SL - m + 1, m, n_t)
        """
        m = self.n_periods
        SL = self.sequence_length
        individual_de = np.zeros((X.shape[0], SL - m + 1, m, self.n_treatments))
        for t in range(self.sequence_length - m + 1):
            for l in range(m - 1, -1, -1):
                individual_de[:, t, l, :] = add_constant(X[:, t+l, :], has_constant='add') @ \
                                                    self.true_hetero_effect[m - 1 - l, :].reshape((self.n_treatments, 1 + self.n_x)).T
        return individual_de


    def compute_treatment_effect(self, intervention_T, baseline_T):
        """
        Args:
            intervention_T: np.ndarray of shape (n_periods,)
            baseline_T: np.ndarray of shape (n_periods,)
            
        Returns:
            np.ndarray: Treatment effect matrix of shape (n_units, sequence_length - n_periods + 1)
        """
        assert intervention_T.shape == (self.n_periods, self.n_treatments)
        assert baseline_T.shape == (self.n_periods, self.n_treatments)
        
        # Simulate counterfactuals
        Y_intervention = self._simulate_counterfactual(intervention_T)
        Y_baseline = self._simulate_counterfactual(baseline_T)
        
        # Calculate treatment effects starting from t=n_periods-1
        return Y_intervention - Y_baseline


    def _simulate_counterfactual(self, T_seq):
        """Simulates outcomes for arbitrary treatment sequence using stored noise"""
        assert T_seq.shape == (self.n_periods, self.n_treatments)
        m = self.n_periods
        #buffer for storing intervened values in time window of length m
        X_ctf_local = np.zeros((self.n_units, m, self.n_x))
        Y_ctf_local = np.zeros((self.n_units, m))
        #expand T_seq
        T_intv = T_seq.reshape(1, self.n_periods, self.n_treatments).repeat(self.n_units, axis = 0)
        #initialize result
        Y_ctf = np.zeros((self.n_units, self.sequence_length - m + 1))
        for t in range(self.sequence_length - self.n_periods + 1):
            #Compute the counterfactual X_ctf and Y_ctf
            for l in range(self.n_periods):
                if l == 0:
                    X_ctf_local[:, 0, :] = self.X_obs[:, t, :]
                else:
                    hetero_multiplier_x = 1 + (X_ctf_local[:, l - 1, :] * self.x_hetero_effect).sum(axis=1)
                    X_ctf_local[:, l, :] = np.expand_dims(hetero_multiplier_x, axis = 1) * np.dot(T_intv[:, l - 1, :], self.Alpha.T) + \
                                        np.dot(X_ctf_local[:, l - 1, :], self.Beta.T) + self.noisex[:, t + l, :]
                hetero_multiplier_y = 1 + (X_ctf_local[:, l, :] * self.y_hetero_effect).sum(axis = 1)
                Y_ctf_local[:, l] = hetero_multiplier_y * np.dot(T_intv[:, l, :], self.epsilon) + \
                                    np.dot(X_ctf_local[:, l, :], self.zeta) + self.noisey[:, t + l]
            Y_ctf[:, t] = Y_ctf_local[:, -1]
        return Y_ctf

    
    def get_processed_data(self, Y, T, X):
        logger.info(f'Processing markovian heterodynamic dataset')
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).double()
        if not isinstance(T, torch.Tensor):
            T = torch.from_numpy(T).double()
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).double()

        prev_T = torch.zeros_like(T)
        prev_T[:, 1:, :] = T[:, :-1, :]

        prev_Y = torch.zeros_like(Y)
        prev_Y[:, 1:] = Y[:, :-1]

        torch.manual_seed(self.params['seed'])
        indices = torch.randperm(Y.shape[0])
        train_size = int(Y.shape[0] * self.train_val_split)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = ProcessedDataset(Y[train_indices], T[train_indices], X[train_indices], 
                                         prev_Y[train_indices], prev_T[train_indices])
        val_dataset = ProcessedDataset(Y[val_indices], T[val_indices], X[val_indices], 
                                         prev_Y[val_indices], prev_T[val_indices])

        return train_dataset, val_dataset
    
    def get_full_dataset(self, Y, T, X):
        logger.info(f'Processing markovian heterodynamic dataset, full dataset, no split')
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).double()
        if not isinstance(T, torch.Tensor):
            T = torch.from_numpy(T).double()
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).double()

        prev_T = torch.zeros_like(T)
        prev_T[:, 1:, :] = T[:, :-1, :]

        prev_Y = torch.zeros_like(Y)
        prev_Y[:, 1:] = Y[:, :-1]
        
        dataset = ProcessedDataset(Y, T, X, prev_Y, prev_T)

        return dataset
        


        




        
