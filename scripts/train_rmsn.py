import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.utils import FilteringMlFlowLogger
from src.models.rmsn import RMSN, RMSNTreatmentPropensityNetwork, RMSNHistoryPropensityNetwork, RMSNEncoderNetwork, RMSNDecoderNetwork 
from src.utils import compute_gt_individual_dynamic_effects
import pickle
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)
OmegaConf.register_new_resolver("times", lambda x, y: x * y, replace = True)
OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace = True)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace = True)

@hydra.main(version_base='1.1', config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))
    seed_everything(args.exp.seed)
    #instantiate dataset pipeline
    data_pipeline = instantiate(args.dataset, _recursive_=True)
    data_pipeline.insert_necessary_args_dml_rnn(args)
    disc_dim, cont_dim = args.dataset.n_treatments_disc, args.dataset.n_treatments_cont


    experiment_name = args.exp.exp_name
    conf_strength = float(data_pipeline.get_confounding_strength())
    n_periods = args.dataset.n_periods
    train_data, val_data = data_pipeline.train_data, data_pipeline.val_data
    
    ####===========Propensity treatment / history network===================####
    mlf_logger_propensity = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
        tracking_uri=args.exp.mlflow_uri, run_name=f"rmsn_conf={conf_strength}_m={n_periods}_propensity")
    
    artifacts_path_propensity = hydra.utils.to_absolute_path(
        mlf_logger_propensity.experiment.get_run(
            mlf_logger_propensity.run_id).info.artifact_uri).replace('mlflow-artifacts:', 'mlruns')
    logger.info(f"Artifacts path : {artifacts_path_propensity}")

    #initialize data loader
    train_loader = DataLoader(train_data, batch_size=args.exp.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.exp.batch_size, shuffle=False)

    #initialize model
    propensity_treatment_network = RMSNTreatmentPropensityNetwork(args)
    
    #callbacks = []
    callbacks = [LearningRateMonitor(logging_interval='epoch')]

    #initialize trainer
    trainer_pt = Trainer(
                max_epochs=args.exp.max_epochs,
                callbacks=callbacks,
                devices=1,
                accelerator=args.exp.accelerator,
                deterministic=True,
                logger = mlf_logger_propensity,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
    logger.info("Train propensity treatment network")
    trainer_pt.fit(propensity_treatment_network, train_loader, val_loader)

    propensity_history_network = RMSNHistoryPropensityNetwork(args)
    traner_ph = Trainer(
                max_epochs=args.exp.max_epochs,
                callbacks=callbacks,
                devices=1,
                accelerator=args.exp.accelerator,
                deterministic=True,
                logger = mlf_logger_propensity,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
    logger.info("Train propensity history network")
    traner_ph.fit(propensity_history_network, train_loader, val_loader)

    
    ####===============Compute stablized weights===================####
    logger.info("Compute stablized weights")



    #####===========Encoder / Decoder network===================####





if __name__ == '__main__':
    main()

