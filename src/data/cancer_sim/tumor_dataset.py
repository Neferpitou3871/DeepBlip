from src.data.cancer_sim.simulation import generate_params, get_standard_params, \
    simulate_factual, simulate_counterfactuals_treatment_seq
from torch.utils.data import Dataset
from omegaconf import DictConfig

#Define the tumor growth dataset class
class TumorGrowthDataset(Dataset):
    def __init__(self, args_dataset:DictConfig):
        """
        Args:
            chemo_coeff: Confounding coefficient of chemotherapy
            radio_coeff: Confounding coefficient of radiotherapy
            num_patients: Number of patients in dataset
            window_size: Used for biased treatment assignment (the number of periods to look back)
            n_periods: Number of periods to intervene treatments
            lag: lag to perform treatment (for now just 0)
            seq_length: Max length of time series
            seed: Random seed
        """
        self.params = args_dataset
        self.chemo_coeff = args_dataset['chemo_coeff']
        self.radio_coeff = args_dataset['radio_coeff']
        self.n_units = args_dataset['n_units']
        self.n_periods = args_dataset['n_periods']
        self.sequence_length = args_dataset['sequence_length']
        self.window_size = args_dataset['window_size']
        self.lag = args_dataset['lag']
        self.seed = args_dataset['seed']

        # Generate parameters for simulation
        self.patient_params = generate_params(self.n_units, self.chemo_coeff, self.radio_coeff, \
                                              self.window_size, self.lag)
        
        