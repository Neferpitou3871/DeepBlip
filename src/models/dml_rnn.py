import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
import numpy as np
from omegaconf import DictConfig
import logging
import mlflow

from src.models.utils_lstm import VariationalLSTM
from src.models.utils import build_phi

logger = logging.getLogger(__name__)
#ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10**8  # ~ 100Mb

class OutcomeHead(nn.Module):
    def __init__(self, hidden_size, fc_hidden_size, dim_outcome=1, dim_outcome_disc=0):
        super().__init__()
        self.dim_outcome = dim_outcome
        self.dim_outcome_disc = dim_outcome_disc
        self.dim_outcome_cont = dim_outcome - dim_outcome_disc
        self.linear1 = nn.Linear(hidden_size, fc_hidden_size)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(fc_hidden_size, dim_outcome)
        self.trainable_params = ['linear1', 'linear2']
    
    def build_outcome(self, hr):
        """
        hr: hidden representation of patient state, shape (b, hr_size)
        returns: outcome, shape (b, dim_outcome)
        """
        x = self.elu(self.linear1(hr))
        outcome = self.linear2(x)
        #first dim_outcome_disc elements need to be transformed to [0, 1] through sigmoid function
        if self.dim_outcome_disc > 0:
            prob = torch.sigmoid(outcome[:, :self.dim_outcome_disc])
            outcome = torch.concat([prob, outcome[:, self.dim_outcome_disc:]], dim = -1)
        return outcome

class Nuisance_Network(LightningModule):

    def __init__(self, args: DictConfig):
        """
        Args:
            args: DictConfig of model hyperparameters
            phi: function ()
        note1: currently only support the case where phi is the current treatment
        note2: currently only support binary type for discrete treatment
        """
        super().__init__()
        dataset_params = args.dataset
        self.model_type = 'nuisace parameter network'
        self.n_treatments = dataset_params['n_treatments']
        self.n_treatments_disc = dataset_params.get('n_treatments_disc', 0)
        self.n_treatments_cont = dataset_params.get('n_treatments_cont', 0)
        assert self.n_treatments == self.n_treatments_disc + self.n_treatments_cont
        self.n_x = dataset_params['n_x']
        self.n_static = dataset_params.get('n_static', 0)
        self.n_periods = dataset_params['n_periods']
        self.static_treatment_policy = torch.tensor(args.model.static_treatment_policy, dtype=torch.float32)
        self.input_size = self.n_treatments + self.n_x + self.n_static + 1
        self.sequence_length = dataset_params['sequence_length']
        self.phi = build_phi(args.model.phi_type)
        self.phi_type = args.model.phi_type
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')
        self.save_hyperparameters(args)

        self._initialize_model(args)
    
    def _initialize_model(self, args: DictConfig):

        self.hidden_size = args.model.hidden_size
        self.hr_size = args.model.hr_size
        self.num_layer = args.model.num_layer
        self.dropout_rate = args.model.dropout_rate
        self.fc_hidden_size_p = args.model.fc_hidden_size_p
        self.fc_hidden_size_q = args.model.fc_hidden_size_q
        self.dim_phi = args.model.dim_phi
        assert self.dim_phi == self.n_treatments

        self.lstm = VariationalLSTM(self.input_size, self.hidden_size, self.num_layer, self.dropout_rate)
        self.hr_output_transformation = nn.Linear(self.hidden_size, self.hr_size)
        self.output_dropout = nn.Dropout(self.dropout_rate)
        self.p_comp_head = nn.ModuleList([
            OutcomeHead(self.hr_size, self.fc_hidden_size_p, 1) for _ in range(self.n_periods)
        ])
        self.q_comp_head = nn.ModuleList([
            nn.ModuleList([
                OutcomeHead(self.hr_size, self.fc_hidden_size_q, self.dim_phi, self.n_treatments_disc) 
                if k >= p else nn.Module() for k in range(self.n_periods)
            ]) for p in range(self.n_periods)
        ])
    

    def build_hr(self, static_features, curr_covariates, prev_treatments, prev_outputs):
        """
        Build hidden representation (patient clinical state)
        """
        #expand static_features along the time dimension (b, n_static) -> (b, L, n_static)
        static_features = static_features.unsqueeze(1).expand(-1, self.sequence_length, -1)   
        x = torch.cat([static_features, curr_covariates, prev_treatments, prev_outputs.unsqueeze(-1)], dim = -1)
        x = self.lstm(x, init_states=None)
        output = self.output_dropout(x)
        hr = nn.ELU()(self.hr_output_transformation(output))
        return hr
    
    def compute_Q(self, current_covariates, curr_treatments):
        """
        Build Q values (target for q compuation heads)
        """
        batch_size = current_covariates.size(0)
        Q = torch.zeros((batch_size, self.sequence_length - self.n_periods + 1, 
                                        self.n_periods, self.n_periods, self.dim_phi), device=self.device)
        if (self.phi_type == 'current_treatment') and (self.static_treatment_policy == torch.zeros(self.static_treatment_policy.size())).all():
            for t in range(self.sequence_length - self.n_periods + 1):
                for l in range(self.n_periods):
                    Q[:, t, l, l : self.n_periods, :] = curr_treatments[:, t + l: t + self.n_periods, :]
        #for t in range(self.sequence_length - self.n_periods + 1):
        #    for d
        return Q

    def forward(self, batch: dict):
        """
        batch info:
        prev_treatments: torch.tensor shape (b, L, n_treatments)
        prev_outpus: torch.tensor shape (b, L, 1)
        returns:
        p_pred_all_steps: torch.tensor shape (b, SL - m + 1, m)
        q_pred_all_steps: torch.tensor shape (b, SL - m + 1, m, m, disc_dim + cont_dim), the discrete dim outputs are in [0, 1]
        """
        prev_outputs = batch['prev_outputs']
        b, L = prev_outputs.size(0), prev_outputs.size(1)
        prev_treatments_disc = batch['prev_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments_cont = batch['prev_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        prev_treatments = torch.cat([prev_treatments_disc, prev_treatments_cont], dim = -1)
        static_features = batch['static_features'] if self.n_static > 0 else torch.zeros((b, 0), device=self.device)
        curr_covariates = batch['curr_covariates']
        prev_outputs = batch['prev_outputs']
        batch_size = prev_treatments.size(0)
        assert prev_treatments.size(1) == self.sequence_length
        p_pred_all_steps = torch.zeros((batch_size, self.sequence_length - self.n_periods + 1, self.n_periods), device = self.device)
        q_pred_all_steps = torch.zeros((batch_size, self.sequence_length - self.n_periods + 1, 
                                        self.n_periods, self.n_periods, self.dim_phi), device=self.device)
        #q_res_all_steps = torch.zeros((batch_size, self.sequence_length - self.n_periods + 1, self.n_periods), device = self.device)
        
        hr = self.build_hr(static_features, curr_covariates, prev_treatments, prev_outputs) # dim = (b, L. hr_size)
        for t in range(self.sequence_length - self.n_periods + 1): # t = 0, 1, ., L - m
            for l in range(self.n_periods): #l = 0, 1, . ., m - 1
                p_pred_all_steps[:, t, l] = self.p_comp_head[l].build_outcome(hr[:, t + l, :]).squeeze(-1)
                #q_res_all_steps[:, t, l] = target_outcome - q_pred_all_
                for j in range(t + l, t + self.n_periods): # j = t+l, ., t+m -1
                    #Q = self.compute_Q(curr_covariates, curr_treatments)
                    q_pred_all_steps[:, t, l, j - t, :] = self.q_comp_head[l][j - t].build_outcome(hr[:, j, :])
        
        return p_pred_all_steps, q_pred_all_steps
    
    def on_train_start(self):
        for par in self.parameters():
            par.requires_grad = True
    
    def training_step(self, batch, batch_ind, optimizer_idx=None):

        curr_outputs = batch['curr_outputs']
        b, L = curr_outputs.size(0), curr_outputs.size(1)
        curr_treatments_disc = batch['curr_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments_cont = batch['curr_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments = torch.cat([curr_treatments_disc, curr_treatments_cont], dim = -1)
        curr_covariates = batch['curr_covariates']
        active_entries = batch['active_entries']

        p_pred_all_steps, q_pred_all_steps = self.forward(batch)
        #separate q_pred_all_steps / q_gt_all_steps to disc and cont
        q_pred_all_steps_disc = q_pred_all_steps[:, :, :, :, :self.n_treatments_disc]
        q_pred_all_steps_cont = q_pred_all_steps[:, :, :, :, self.n_treatments_disc:]

        Q_gt_all_steps = self.compute_Q(curr_covariates, curr_treatments)
        Q_gt_all_steps_disc = Q_gt_all_steps[:, :, :, :, :self.n_treatments_disc]
        Q_gt_all_steps_cont = Q_gt_all_steps[:, :, :, :, self.n_treatments_disc:]
        p_target = curr_outputs[:, self.n_periods - 1:].unsqueeze(-1).expand(-1, -1, self.n_periods)

        p_mse = (F.mse_loss(p_pred_all_steps, p_target, reduction='none') * active_entries).mean(dim = (0, 1))
        for i in range(p_mse.shape[0]):
            self.log(f'p{i}_mse', p_mse[i], on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        
        upper_traingle_mask = torch.triu(torch.ones((self.n_periods, self.n_periods), dtype=torch.bool, device = self.device))
        q_bce, q_mse = 0, 0
        if self.n_treatments_disc > 0:
            #compute the binary cross entropy loss for the discrete treatments (Since we only implement for binary treatment)
            q_pred_disc = q_pred_all_steps_disc[:, :, upper_traingle_mask, :]
            q_target_disc = Q_gt_all_steps_disc[:, :, upper_traingle_mask, :]
            q_bce = F.binary_cross_entropy(q_pred_disc, q_target_disc, reduction = 'none').mean(dim = (0, 1, 2))
            self.log('q_bce', q_bce, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        if self.n_treatments_cont > 0:
            q_pred_cont = q_pred_all_steps_cont[:, :, upper_traingle_mask, :]
            q_target_cont = Q_gt_all_steps_cont[:, :, upper_traingle_mask, :]
            q_mse = F.mse_loss(q_pred_cont, q_target_cont, reduction='none').mean(dim = (0, 1, 2))
            self.log('q_mse', q_mse, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        
        loss = q_bce + q_mse + p_mse.mean()
        self.log('train_loss', loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_ind):

        curr_outputs = batch['curr_outputs']
        b, L = curr_outputs.size(0), curr_outputs.size(1)
        curr_treatments_disc = batch['curr_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments_cont = batch['curr_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments = torch.cat([curr_treatments_disc, curr_treatments_cont], dim = -1)
        curr_covariates = batch['curr_covariates']
        active_entries = batch['active_entries']

        p_pred_all_steps, q_pred_all_steps = self.forward(batch)
        Q_gt_all_steps = self.compute_Q(curr_covariates, curr_treatments)
        p_target = curr_outputs[:, self.n_periods - 1:].unsqueeze(-1).expand(-1, -1, self.n_periods)

        #Do similar thing as in training_step
        p_mse = (F.mse_loss(p_pred_all_steps, p_target, reduction='none') * active_entries).mean(dim = (0, 1))
        for i in range(p_mse.shape[0]):
            self.log(f'val_p{i}_mse', p_mse[i], on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        
        upper_traingle_mask = torch.triu(torch.ones((self.n_periods, self.n_periods), dtype=torch.bool, device = self.device))
        q_bce, q_mse = 0, 0
        if self.n_treatments_disc > 0:
            q_pred_disc = q_pred_all_steps[:, :, upper_traingle_mask, :self.n_treatments_disc]
            q_target_disc = Q_gt_all_steps[:, :, upper_traingle_mask, :self.n_treatments_disc]
            q_bce = F.binary_cross_entropy(q_pred_disc, q_target_disc, reduction = 'none').mean(dim = (0, 1))
            for i in range(q_bce.shape[0]):
                self.log(f'val_q{i}[.]_bce', q_bce[i], on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        if self.n_treatments_cont > 0:
            q_pred_cont = q_pred_all_steps[:, :, upper_traingle_mask, self.n_treatments_disc:]
            q_target_cont = Q_gt_all_steps[:, :, upper_traingle_mask, self.n_treatments_disc:]
            q_mse = F.mse_loss(q_pred_cont, q_target_cont, reduction='none').mean(dim = (0, 1))
            for i in range(q_mse.shape[0]):
                self.log(f'val_q{i}[.]_mse', q_mse[i], on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)

        #p_mse = F.mse_loss(p_pred_all_steps, p_target, reduction='none').mean(dim=(0, 1))
        #q_mse = F.mse_loss(q_pred_all_steps, Q_gt_all_steps, reduction='none').mean(dim=(0, 1, -1))

        #for i in range(p_mse.shape[0]):
        #    self.log(f'val_p{i}_mse', p_mse[i], on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        #for i in range(q_mse.shape[0]):
        #    for j in range(i, q_mse.shape[1]):
        #        self.log(f"val_q{j}{i}_mse", q_mse[i][j], on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        
        loss = p_mse.mean() + q_mse + q_bce
        self.log('val_loss', loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Select optimizer based on config
        opt_args = self.hparams.model.optimizer
        optimizer_cls = opt_args['optimizer_cls']
        if optimizer_cls.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=opt_args.learning_rate, weight_decay=opt_args.weight_decay)
        elif optimizer_cls.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=opt_args.learning_rate, weight_decay=opt_args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_cls.lower()}")

        return optimizer
    
    def predict_step(self, batch, batch_idx):
        """
        predict_step returns the residuals, not the predicted p and q!
        res_Y shape: (b, SL - n_period + 1, n_period)
        res_T shape: (b, SL - n_period + 1, n_period, n_period, n_treatments)
        """
        curr_outputs = batch['curr_outputs']
        b, L = curr_outputs.size(0), curr_outputs.size(1)
        curr_treatments_disc = batch['curr_treatments_disc'] if self.n_treatments_disc > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments_cont = batch['curr_treatments_cont'] if self.n_treatments_cont > 0 else torch.zeros((b, L, 0), device=self.device)
        curr_treatments = torch.cat([curr_treatments_disc, curr_treatments_cont], dim = -1)
        curr_covariates = batch['curr_covariates']

        p_pred_all_steps, q_pred_all_steps = self.forward(batch)
        Q_gt_all_steps = self.compute_Q(curr_covariates, curr_treatments)
        p_target = curr_outputs[:, self.n_periods - 1:].unsqueeze(-1).expand(-1, -1, self.n_periods)

        assert (p_target.shape == p_pred_all_steps.shape) and (q_pred_all_steps.shape == Q_gt_all_steps.shape)
        res_Y = p_target - p_pred_all_steps
        res_T = Q_gt_all_steps - q_pred_all_steps

        return res_Y, res_T
    


class DynamicEffect_estimator(LightningModule):

    def __init__(self, args: DictConfig, true_effect: np.array = None):
        """
        Args:
            args: DictConfig of model hyperparameters
            true_effect: only effective when using the linear dataset, otherwise should be None
            phi: function ()
        """
        super().__init__()
        dataset_params = args.dataset
        self.model_type = 'Blip Parameter estimator network'
        self.n_treatments = dataset_params['n_treatments']
        self.n_x = dataset_params['n_x']
        self.n_periods = dataset_params['n_periods']
        self.input_size = self.n_treatments + self.n_x + 1
        self.sequence_length = dataset_params['sequence_length']
        self.true_effect = np.flip(true_effect, axis = 0) if true_effect is not None else None
        self.loss_type = args.model.loss_type
        if self.loss_type == 'moment':
            self.moment_order = int(args.model.moment_order)
        self.double_opt = args.model.double_opt
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')
        self.save_hyperparameters(args)
        self.args = args

        self._initialize_model(args)
    
    def _initialize_model(self, args: DictConfig):

        self.hidden_size = args.model.hidden_size
        self.hr_size = args.model.hr_size
        self.num_layer = args.model.num_layer
        self.dropout_rate = args.model.dropout_rate
        self.fc_hidden_size_psi = args.model.fc_hidden_size_psi
        self.dim_phi = args.model.dim_phi
        assert self.dim_phi == self.n_treatments

        self.lstm = VariationalLSTM(self.input_size, self.hidden_size, self.num_layer, self.dropout_rate)
        self.hr_output_transformation = nn.Linear(self.hidden_size, self.hr_size)
        self.output_dropout = nn.Dropout(self.dropout_rate)
        self.parameter_head = nn.ModuleList([
            OutcomeHead(self.hr_size, self.fc_hidden_size_psi, self.dim_phi) for _ in range(self.n_periods)
        ])
    

    def build_hr(self, curr_covariates, prev_treatments, prev_outputs):
        """
        Build hidden representation (patient clinical state)
        """
        x = torch.cat([curr_covariates, prev_treatments, prev_outputs.unsqueeze(-1)], dim = -1)
        x = self.lstm(x, init_states=None)
        output = self.output_dropout(x)
        hr = nn.ELU()(self.hr_output_transformation(output))
        return hr

    def forward(self, batch: dict):
        """
        batch info:
        prev_treatments: torch.tensor shape (b, L, n_treatments)
        prev_outpus: torch.tensor shape (b, L, 1)
        returns: estimated parameter (b, SL - m + 1, n_periods, n_treatments)
        """
        prev_treatments = batch['prev_treatments']
        curr_treatments = batch['curr_treatments']
        curr_covariates = batch['curr_covariates']
        prev_outputs = batch['prev_outputs']
        batch_size = prev_treatments.size(0)
        
        param_pred_all_steps = torch.zeros((batch_size, self.sequence_length - self.n_periods + 1, self.n_periods, self.n_treatments), device = self.device)
        
        hr = self.build_hr(curr_covariates, prev_treatments, prev_outputs) # dim = (b, L. hr_size)
        for t in range(self.sequence_length - self.n_periods + 1): # t = 0, 1, ., L - m
            for l in range(self.n_periods): #l = 0, 1, . ., m - 1
                param_pred_all_steps[:, t, l, :] = self.parameter_head[l].build_outcome(hr[:, t, :])

        return param_pred_all_steps
    
    def predict_step(self, batch:dict, batch_idx = None):
        return self(batch)
    
    def OLS_loss(self, param_pred_all_steps, res_Y_all_steps, res_T_all_steps):
        #res_Y_all_steps (b, SL - n_period + 1, n_periods)
        #res_T_all_steps (b, SL - n_period + 1, n_period, n_period, n_treatments)
        #param_pred_all_steps (b, SL - n_period + 1, n_periods, n_treatments)
        #Compute the sum of all the n_periods squared loss as the final OLS_loss, for simultaneous optimization
        #need to make sure res_Y and res_T are detatched
        m = self.n_periods
        L = torch.zeros((m,), device = self.device)
        #error_all_steps = torch.zeros(res_Y_all_steps.shape, device = self.device)
        error_list = []
        for k in range(m - 1, -1, -1):
            if k == m - 1:
                error_k = (res_Y_all_steps[:, :, m - 1] - (param_pred_all_steps[:, :, m - 1, :] * res_T_all_steps[:, :, m - 1, m - 1, :]).sum(dim = -1))
            else:
                error_k = res_Y_all_steps[:, :, k] - (param_pred_all_steps[:, :, k:, :] * res_T_all_steps[:, :, k, k:, :]).sum(dim = (-1, -2))
            error_list.append(error_k)
        error_list.reverse()
        error_all_steps = torch.stack(error_list, dim = 2)
        L = error_all_steps.pow(2).mean(dim = (0, 1))

        return L, error_all_steps
    
    def moment_loss(self, param_pred_all_steps, param_pred_all_steps_detach, res_Y_all_steps, res_T_all_steps):
        """
        Optimizing directly to reduce the moment condition equation to zero
        """
        m = self.n_periods
        L = torch.zeros((m,), device = self.device)
        ord = self.moment_order
        res_list = []
        if (self.double_opt) and (param_pred_all_steps_detach is not None): #apply double way optimization, one way is detached, one way with grad
            for k in range(m - 1, -1, -1):
                if k == m - 1:
                    error_k = res_Y_all_steps[:, :, k] - (param_pred_all_steps[:, :, m - 1, :] * res_T_all_steps[:, :, m - 1, m -1, :]).sum(dim = -1)
                else:
                    error_k = res_Y_all_steps[:, :, k] - (param_pred_all_steps_detach[:, :, k+1:, :] * res_T_all_steps[:, :, k, k+1:, :]).sum(dim = (-1, -2)) -\
                                                        (param_pred_all_steps[:, :, k, :] * res_T_all_steps[:, :, k, k, :]).sum(dim = -1)
                prod = error_k.unsqueeze(-1) * res_T_all_steps[:, :, k, k, :]
                if ord == 1:
                    res = torch.abs(prod.mean(dim=0)).mean()
                elif ord == 2:
                    res = (prod.mean(dim = 0) ** 2).mean()
                else:
                    raise ValueError(f"illegal ord:{ord}")
                res_list.append(res)

        else: #else average all the moment errors
            for k in range(m - 1, -1, -1):
                error_k = res_Y_all_steps[:, :, k] - (param_pred_all_steps[:, :, k:, :] * res_T_all_steps[:, :, k, k:, :]).sum(dim = (-1, -2))
                prod = error_k.unsqueeze(-1) * res_T_all_steps[:, :, k, k, :] # prod shape [B, SL - m + 1, n_t]
                if ord == 1:
                    res = torch.abs(prod.mean(dim=0)).mean()
                elif ord == 2:
                    res = (prod.mean(dim = 0) ** 2).mean()
                else:
                    raise ValueError(f"illegal ord:{ord}")
                res_list.append(res)
        res_list.reverse()
        moment_losses = torch.stack(res_list)

        return moment_losses
        
    
    def on_train_start(self):
        for par in self.parameters():
            par.requires_grad = True
            
    def training_step(self, batch, batch_idx):

        res_Y_all_steps = batch['residual_Y']
        res_T_all_steps = batch['residual_T']

        param_pred_all_steps = self(batch)
        param_pred_all_steps_detach = self(batch).detach() if self.double_opt else None

        if self.loss_type == 'OLS':
            L, error_all_steps = self.OLS_loss(param_pred_all_steps, res_Y_all_steps, res_T_all_steps)

            for i in range(L.shape[0]):
                self.log(f'train_L{i}_mse', L[i], on_epoch=True, on_step=False, sync_dist=True, prog_bar=False)
            loss = L.mean()
        elif self.loss_type == 'moment':
            moment_losses = self.moment_loss(param_pred_all_steps, param_pred_all_steps_detach, res_Y_all_steps, res_T_all_steps)
            loss = moment_losses.mean()
            for i in range(moment_losses.shape[0]):
                self.log(f'train_norm{i}', moment_losses[i].detach().cpu(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=False)
        else:
            raise ValueError(f"illegal loss type: {self.loss_type}")
            
        self.log('train_loss_param', loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)

        if self.true_effect is not None:
            metrics = self.true_effect_rmse(param_pred_all_steps)
            for key in metrics:
                self.log('train_' + key, metrics[key], on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        
        return loss
    

    def validation_step(self, batch, batch_idx):

        res_Y_all_steps = batch['residual_Y']
        res_T_all_steps = batch['residual_T']

        param_pred_all_steps = self(batch)
        if self.loss_type == 'OLS':
            L, error_all_steps = self.OLS_loss(param_pred_all_steps, res_Y_all_steps, res_T_all_steps)

            for i in range(L.shape[0]):
                self.log(f'val_L{i}_mse', L[i], on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
            loss = L.mean()
        elif self.loss_type == 'moment':
            moment_losses = self.moment_loss(param_pred_all_steps, None, res_Y_all_steps, res_T_all_steps)
            loss = moment_losses.mean()
            for i in range(moment_losses.shape[0]):
                self.log(f'val_norm{i}', moment_losses[i].detach().cpu(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=False)
        else:
            raise ValueError(f"illegal loss type: {self.loss_type}")

        self.log('val_loss_param', loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)

        if self.true_effect is not None:
            metrics = self.true_effect_rmse(param_pred_all_steps)
            for key in metrics:
                self.log('val_' + key, metrics[key], on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss
    
    def true_effect_moment_norm(self, dataloader):

        result = list()
        K = self.sequence_length - self.n_periods + 1
        true_effect = torch.from_numpy(self.true_effect.copy()).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                res_Y_all_steps = batch['residual_Y']
                res_T_all_steps = batch['residual_T']
                true_effect_expanded = true_effect.expand(res_Y_all_steps.shape[0], K, self.n_periods, self.n_treatments)
                moment_losses = self.moment_loss(true_effect_expanded, None, res_Y_all_steps, res_T_all_steps)
                result.append(moment_losses)
        loss = torch.stack(result).mean(dim = 0)
        return loss
        
    def log_true_effect_moment_norm(self, dataloader, mlf_logger = None):
        logger.info("Computing the moment loss on true effect parameter")
        loss = self.true_effect_moment_norm(dataloader)
        if mlf_logger is not None:
            mlflow.set_tracking_uri(self.args.exp.mlflow_uri)
            with mlflow.start_run(run_id=mlf_logger.run_id, nested=True):
                for i in range(self.n_periods):
                    mlflow.log_metric(
                        run_id=mlf_logger.run_id,
                        key=f"true_effect_moment_loss_step{i}",
                        value=loss[i].item()
                    )
        logger.info("metrics of true effect logged")


    
    def true_effect_rmse(self, param_pred_all_steps):
        """
        Compute the rmse between preditcted param and true effect
        """
        param_pred_np = param_pred_all_steps.detach().cpu().numpy()

        D, K, m, n_t = param_pred_np.shape
        param_preds = param_pred_np.reshape(D * K, m, n_t)

        metrics = {}
        for i in range(m):
            preds_i = param_preds[:, i, :]
            true_i = self.true_effect[i, :]
            diff = preds_i - true_i
            rmse_i = np.sqrt(np.mean(diff ** 2))
            metrics[f'param_rmse{i}'] = rmse_i
        return metrics
    
    def predict_treatment_effect(self, dynamic_effects, T_intv_disc, T_intv_cont, T_base_disc, T_base_cont):
        """
        Estimate treatment effect E[Y(T_intv) - Y^(T_base)]
        Args:
            dynamic_effects: torch Tensor (N, SL - m + 1, m, n_t)
            T_intv, T_base: numpy array / Tensor (m, n_t)
        Returns:
            treatment effect: numpy array (N, SL - m + 1)
        """
        T_intv = self._combine_disc_cont(T_intv_disc, T_intv_cont)
        T_base = self._combine_disc_cont(T_base_disc, T_base_cont)
        de = dynamic_effects.detach().cpu().numpy()
        T_diff = (T_intv - T_base).reshape((1, 1, T_intv.shape[0], T_intv.shape[1]))
        return (de * T_diff).sum((-2, -1))

    def configure_optimizers(self):
        # Select optimizer based on config
        opt_args = self.hparams.model.optimizer
        optimizer_cls = opt_args['optimizer_cls']
        if optimizer_cls.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=opt_args.learning_rate, weight_decay=opt_args.weight_decay)
        elif optimizer_cls.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=opt_args.learning_rate, weight_decay=opt_args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_cls.lower()}")

        return optimizer
    
    def _combine_disc_cont(self, T_disc, T_cont):
        """
        Combine discrete and continuous treatments
        """
        if T_disc is None:
            return T_cont
        elif T_cont is None:
            return T_disc
        else:
            assert T_disc.shape[:-1] == T_cont.shape[:-1]
            return torch.cat([T_disc, T_cont], dim = -1)

        




    


        





    







