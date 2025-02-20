import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.data.mimic.semi_synthetic_dataset import SyntheticOutcomeGenerator, SyntheticTreatment
from src.data.mimic.load_data import load_mimic3_data_raw
from src.data.mimic.utils import sigmoid, SplineTrendsMixture
from src.data.base_dataset_pipeline import BaseDatasetPipeline, ProcessedDataset
#import List
from typing import List, Dict
import logging
logger = logging.getLogger(__name__)
from joblib import Parallel, delayed
from tqdm import tqdm

#Create a dataset pipeline for MIMIC-III semisynthetic data '
class MIMICSemiSyntheticDataPipeline(BaseDatasetPipeline):

    def __init__(self,
                 path: str,
                 min_seq_length: int,
                 max_seq_length: int,
                 max_number: int,
                 n_periods: int,
                 vital_list: List[str],
                 static_list: List[str],
                 synth_outcome: SyntheticOutcomeGenerator,
                 synth_treatments_list: List[SyntheticTreatment],
                 treatment_outcomes_influence: Dict[str, list[str]],
                 autoregressive: bool = True,
                 parallel: bool = False,
                 split = {'val': 0.15, 'test': 0.15},
                 seed=2025,
                 **kwargs):
        
        super().__init__(seed = seed, 
                         n_treatments = len(synth_treatments_list),
                         n_treatments_cont= 0,
                         n_treatments_disc= len(synth_treatments_list),
                         n_units = max_number,
                         n_periods = n_periods,
                         sequence_length= max_seq_length,
                         split = split,
                         kwargs = kwargs)
        
        self.synth_outcome = synth_outcome
        self.synthetic_outcomes = [synth_outcome]
        self.synthetic_treatments = synth_treatments_list
        self.treatment_outcomes_influence = treatment_outcomes_influence
        self.seed = seed
        self.path = path,
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.vital_cols = vital_list
        self.static_list = static_list
        self.autoregressive = autoregressive
        self.parallel = parallel
        self.name = 'MIMIC-III Semi-Synthetic Data Pipeline'

        self.all_vitals, self.static_features = load_mimic3_data_raw(path, min_seq_length, max_seq_length, 
                                                                     max_number=max_number, 
                                                                     vital_list = vital_list,
                                                                     static_list = static_list,
                                                                     data_seed = seed)
        #Reset subject_id from 0 to n_units - 1
        self._remap_index()

        self.treatment_cols = [t.treatment_name for t in self.synthetic_treatments]
        self.outcome_col = synth_outcome.outcome_name
        self.treatment_options = [0., 1.] #currently only support binary treatments

        #Simulate untreated outcome Z for all the data, appending y_endo, y_exog, 
        # y_untreated and y (as placeholder for further compuation)
        self.synth_outcome.simulate_untreated(self.all_vitals, self.static_features)

        for treatment in self.synthetic_treatments:
            self.all_vitals[treatment.treatment_name] = 0.0
        self.all_vitals['fact'] = np.nan
        self.all_vitals.loc[(slice(None), 0), 'fact'] = 1.0
        user_sizes = self.all_vitals.groupby(level='subject_id', sort=False).size()

        logger.info(f'Simulating factual treatments and applying them to outcomes.')
        par = Parallel(n_jobs=4, backend='loky')
        seeds = np.random.randint(0, 10000, size=len(self.static_features))
        if parallel:
            self.all_vitals = par(delayed(self.treat_patient_factually)(patient_ix, seed)
                            for patient_ix, seed in tqdm(zip(self.static_features.index, seeds), total=len(self.static_features)))
        else:
            #Process all the patients sequentially
            self.all_vitals = [self.treat_patient_factually(patient_ix, seed) for patient_ix, seed in \
                                        tqdm(zip(self.static_features.index, seeds), total=len(self.static_features))]
        logger.info('Concatenating all the trajectories together.')
        #Each single patient dataframe has new columns: 
        # ['y1_exog', 'y1_endo', 'y1_untreated', 'y1', 'y2_exog'.., 'y2', 'fact', 't1', 't2']
        self.all_vitals = pd.concat(self.all_vitals, keys=self.static_features.index)
        #Restore the name of subjec_id to the first level index
        self.all_vitals.index = self.all_vitals.index.set_names(['subject_id', 'hours_in'])

        # Padding with nans
        self.all_vitals = self.all_vitals.unstack(fill_value=np.nan, level=0).stack(dropna=False).swaplevel(0, 1).sort_index()
        
        #Conversion to numpy arrays
        self.static_features = self.static_features.sort_index()
        self.static_features = self.static_features.values
        self.treatments = self.all_vitals[self.treatment_cols].fillna(0.0).values.reshape((-1, max(user_sizes),
                                                                                      len(self.treatment_cols)))
        self.vitals_np = self.all_vitals[self.vital_cols].fillna(0.0).values.reshape((-1, max(user_sizes), len(self.vital_cols)))
        self.outcome_unscaled = self.all_vitals[self.outcome_col].fillna(0.0).values.reshape((-1, max(user_sizes)))
        self.outcome_scaled = self._get_scaled_outcome()
        self.active_entries = (~self.all_vitals.isna().all(1)).astype(float)
        self.active_entries = self.active_entries.values.reshape((-1, max(user_sizes), 1))
        self.user_sizes = np.squeeze(self.active_entries.sum(1))

        logger.info(f'Shape of exploded vitals: {self.vitals_np.shape}.')

        #split data, self.train_data, self.train_index, self.index
        self._split_data()
        self.train_data = self.get_torch_dataset(subset='train')
        self.val_data = self.get_torch_dataset(subset='val')
        self.test_data = self.get_torch_dataset(subset='test')
        logger.info('Data pipeline initialized.')

    def _remap_index(self):
        """
        Remap the index of self.all_vitals to start from 0 to self.n_units - 1,
        set the subject_id of self.all_vitals and self.static_features to be the same, namely 0, 1, 2, ..., self.n_units - 1
        """
        assert self.all_vitals is not None
        assert self.static_features is not None
        self.index_mapping = self.all_vitals.index.get_level_values('subject_id').unique()
        assert self.n_units == len(self.index_mapping)
        mapping = {old_id: new_id for new_id, old_id in enumerate(self.index_mapping)}
        new_index = pd.MultiIndex.from_tuples(
            [(mapping[old_id], traj_id) for old_id, traj_id in self.all_vitals.index],
            names = ['subject_id', 'hours_in']
        )
        self.all_vitals.index = new_index
        self.static_features.index = range(self.n_units)

    
    def _split_data(self):
        """
            split data into train, val, test by self.split, store index in self.index, self.train_index, self.val_index, self.test_index
        """
        #first assert that the data is sorted by subject_id and from 0 to self.n_units - 1
        assert self.all_vitals.index.get_level_values('subject_id').is_monotonic_increasing
        assert self.all_vitals.index.get_level_values('subject_id').min() == 0
        assert self.all_vitals.index.get_level_values('subject_id').max() == self.n_units - 1
        #get the index of self.all_vitals
        index = np.arange(self.n_units)
        np.random.shuffle(index)
        train_pos = int(self.n_units * (1 - self.val_split - self.test_split))
        val_pos = int(self.n_units * (1 - self.test_split))
        train_index, val_index, test_index = index[:train_pos], index[train_pos:val_pos], index[val_pos:]
        self.train_index, self.val_index, self.test_index = train_index, val_index, test_index
        self.index = {'train': train_index, 'val': val_index, 'test': test_index}
        logger.info(f'Split data into train, val, test by {self.index}.')

        return
    
    def _get_scaled_outcome(self):
        """
        Scale the outcome to standard normal distribution.
        """
        assert self.outcome_unscaled is not None
        logger.info("Scaling outcomes to standard normal distribution. (only along the first axis (patients))")
        self.scaling_params = {
            'mean': self.outcome_unscaled.mean(axis = 0, keepdims = True),
            'std': self.outcome_unscaled.std(axis = 0, keepdims = True)
        }
        assert (self.scaling_params['std'] == 0.0).sum() == 0
        assert self.scaling_params['mean'].shape == (1, self.outcome_unscaled.shape[1])
        outcome_scaled = (self.outcome_unscaled - self.scaling_params['mean']) / self.scaling_params['std']
        return outcome_scaled
    
    def get_torch_dataset(self, subset='train') -> ProcessedDataset:
        """
        Get the processed dataset in the form of torch.utils.data.Dataset.
        Args:
            subset (str, optional): Subset of the data to get. Defaults to 'train'.
        Returns:
            ProcessedDataset: Processed dataset.
        """
        if subset == 'train':
            index = self.train_index
        elif subset == 'val':
            index = self.val_index
        elif subset == 'test':
            index = self.test_index
        else:
            raise ValueError(f'Invalid subset: {subset}.')

        #start from t = 1, transform to torch tensor
        Y = torch.from_numpy(self.outcome_scaled[index][:, 1:])
        prev_Y = torch.from_numpy(self.outcome_scaled[index][:, :-1])
        T_disc = torch.from_numpy(self.treatments[index][:, 1:, :])
        T_disc_prev = torch.from_numpy(self.treatments[index][:, :-1, :])
        X_static = torch.from_numpy(self.static_features[index])
        X_dynamic = torch.from_numpy(self.vitals_np[index][:, 1:])
        active_entries = torch.from_numpy(self.active_entries[index][:, 1:])

        return ProcessedDataset(Y = Y, prev_Y = prev_Y,
                                T_disc = T_disc, T_cont = None, T_disc_prev = T_disc_prev, T_cont_prev = None,
                                X_staic = X_static, X_dynamic = X_dynamic, 
                                active_entries = active_entries, subset_name = subset)

    def treat_patient_factually(self, patient_ix: int, seed: int):
        """
        Generate factually treated outcomes for a patient.
            patient_ix (int): Index of the patient in the dataset.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        Returns:
            pandas.DataFrame: DataFrame containing the patient's data with factually treated outcomes.
        Note:
            The treatment at hour index t is based on the information of time [t - window, t]
            And the treatment value assigned as t_prev at hour index t + 1, which means at hour 0 no treatment is applied.
            The treatment at hour index t affects the outcome starting from t + 1 (has a limited window of effect)
        """
        patient_df = self.all_vitals.loc[patient_ix].copy()
        rng = np.random.RandomState(seed)
        curr_treatment_cols = [f'{treatment.treatment_name}' for treatment in self.synthetic_treatments]

        for t in range(len(patient_df)):

            # Sampling treatments, based on previous factual outcomes
            treat_probas, treat_flags = self._sample_treatments_from_factuals(patient_df, t, rng)

            if t < max(patient_df.index.get_level_values('hours_in')):
                # Setting factuality flags
                patient_df.loc[t + 1, 'fact'] = 1.0

                # Setting factual sampled treatments
                patient_df.loc[t + 1, curr_treatment_cols] = {t: v for t, v in treat_flags.items()}

                # Treatments applications
                if sum(treat_flags.values()) > 0:

                    # Treating each outcome separately
                    for outcome in self.synthetic_outcomes:
                        common_treatment_range, future_outcomes = self._combined_treating(patient_df, t, outcome, treat_probas,
                                                                                          treat_flags)
                        patient_df.loc[common_treatment_range, f'{outcome.outcome_name}'] = future_outcomes

        return patient_df
    
    def _sample_treatments_from_factuals(self, patient_df, t, rng=np.random.RandomState(None)):
            """
            Sample treatment for patient_df and time-step t
            Args:
                patient_df: DataFrame of patient
                t: Time-step
                rng: Random numbers generator (for parallelizing)

            Returns: Propensity scores, sampled treatments
            """
            factual_patient_df = patient_df[patient_df.fact.astype(bool)]
            treat_probas = {treatment.treatment_name: treatment.treatment_proba(factual_patient_df, t) for treatment in
                            self.synthetic_treatments}
            treatment_sample = {treatment_name: rng.binomial(1, treat_proba)[0] for treatment_name, treat_proba in
                                treat_probas.items()}
            return treat_probas, treatment_sample
    
    def _combined_treating(self, patient_df, t, outcome: SyntheticOutcomeGenerator, treat_probas: dict, treat_flags: dict):
        """
        Combing application of treatments
        Args:
            patient_df: DataFrame of patient
            t: Time-step
            outcome: Outcome to treat
            treat_probas: Propensity scores
            treat_flags: Treatment application flags

        Returns: Combined effect window, combined treated outcome
        """
        treatment_ranges, treated_future_outcomes = [], []
        influencing_treatments = self.treatment_outcomes_influence[outcome.outcome_name]
        influencing_treatments = \
            [treatment for treatment in self.synthetic_treatments if treatment.treatment_name in influencing_treatments]

        for treatment in influencing_treatments:
            treatment_range, treated_future_outcome = \
                treatment.get_treated_outcome(patient_df, t, outcome.outcome_name, treat_probas[treatment.treatment_name],
                                              bool(treat_flags[treatment.treatment_name]))

            treatment_ranges.append(treatment_range)
            treated_future_outcomes.append(treated_future_outcome)

        common_treatment_range, future_outcomes = SyntheticTreatment.combine_treatments(
            treatment_ranges,
            treated_future_outcomes,
            np.array([bool(treat_flags[treatment.treatment_name]) for treatment in influencing_treatments])
        )
        return common_treatment_range, future_outcomes
    

    def plot_timeseries(self, n_patients=5, mode='factual', seed = None):
        """
        Plotting patient trajectories
        Args:
            n_patients: Number of trajectories
            mode: factual / counterfactual
        """
        fig, ax = plt.subplots(nrows=4 * len(self.synthetic_outcomes) + len(self.synthetic_treatments), ncols=1, figsize=(15, 30))
        #Sample random patients
        if seed is not None:
            np.random.seed(seed)
        patient_ixs = np.random.choice(np.arange(self.n_units), n_patients, replace=False)
        for i, patient_ix in enumerate(patient_ixs):
            ax_ind = 0
            factuals = self.all_vitals.fillna(0.0).fact.astype(bool)
            for outcome in self.synthetic_outcomes:
                outcome_name = outcome.outcome_name
                ax[ax_ind].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_exog'].
                                groupby('hours_in').head(1).values)
                ax[ax_ind + 1].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_endo'].
                                    groupby('hours_in').head(1).values)
                ax[ax_ind + 2].plot(self.all_vitals[factuals].loc[patient_ix, f'{outcome_name}_untreated'].
                                    groupby('hours_in').head(1).values)
                if mode == 'factual':
                    ax[ax_ind + 3].plot(self.all_vitals.loc[patient_ix, outcome_name].values)
                elif mode == 'counterfactual':
                    color = next(ax[ax_ind + 3]._get_lines.prop_cycler)['color']
                    ax[ax_ind + 3].plot(self.all_vitals[factuals].loc[patient_ix, outcome_name].
                                        groupby('hours_in').head(1).index.get_level_values(1),
                                        self.all_vitals[factuals].loc[patient_ix, outcome_name].
                                        groupby('hours_in').head(1).values, color=color)
                    ax[ax_ind + 3].scatter(self.all_vitals.loc[patient_ix, outcome_name].index.get_level_values(1),
                                           self.all_vitals.loc[patient_ix, outcome_name].values, color=color, s=2)
                    # for traj_ix in self.all_vitals.loc[patient_ix].index.get_level_values(0):
                    #     ax[ax_ind + 3].plot(self.all_vitals.loc[(patient_ix, traj_ix), outcome_name].index,
                    #                         self.all_vitals.loc[(patient_ix, traj_ix), outcome_name].values, color=color,
                    #                         linewidth=0.05)

                ax[ax_ind].set_title(f'{outcome_name}_exog')
                ax[ax_ind + 1].set_title(f'{outcome_name}_endo')
                ax[ax_ind + 2].set_title(f'{outcome_name}_untreated')
                ax[ax_ind + 3].set_title(f'{outcome_name}')
                ax_ind += 4

            for treatment in self.synthetic_treatments:
                treatment_name = treatment.treatment_name
                ax[ax_ind].plot(self.all_vitals[factuals].loc[patient_ix, f'{treatment_name}'].
                                groupby('hours_in').head(1).values + 2 * i)
                ax[ax_ind].set_title(f'{treatment_name}')
                ax_ind += 1

        fig.suptitle(f'Time series from {self.name}', fontsize=16)
        plt.show()


    
    