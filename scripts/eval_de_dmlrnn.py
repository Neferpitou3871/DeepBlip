import sys,os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import hydra
import torch
import numpy as np
from src.models.deepblip_rnn import Nuisance_Network, DynamicEffect_estimator
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
#import lightning trainer
from pytorch_lightning import Trainer

@hydra.main(version_base='1.1', config_name=f'config.yaml', config_path='../config/')
def main():