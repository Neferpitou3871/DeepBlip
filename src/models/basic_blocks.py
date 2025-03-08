import torch
import torch.nn as nn
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


class OutcomeHead_GRNN(nn.Module):
    """Used by G_RNN"""

    def __init__(self, seq_hidden_units, hr_size, fc_hidden_units, dim_treatments, dim_outcome):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.hr_size = hr_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome

        self.linear1 = nn.Linear(self.hr_size + self.dim_treatments,
                                 self.fc_hidden_units)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(self.fc_hidden_units, self.dim_outcome)
        self.trainable_params = ['linear1', 'linear2']

    def build_outcome(self, hr, current_treatment):
        x = torch.cat((hr, current_treatment), dim=-1)
        x = self.elu(self.linear1(x))
        outcome = self.linear2(x)
        return outcome