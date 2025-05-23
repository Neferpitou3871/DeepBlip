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
from src.data.synthetic_dataset import MarkovianHeteroDynamicDataset
from src.models.deepblip import Nuisance_Network, DynamicEffect_estimator
from src.models.dml import HeteroDynamicPanelDML
from src.models.utils import evaluate_nuisance_mse, plot_residual_distribution, plot_param_est_distribution, transform_residual_data
from src.utils import log_params_from_omegaconf_dict
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

@hydra.main(version_base='1.1', config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))
    seed_everything(args.exp.seed)
    #create dataset
    hddataset = MarkovianHeteroDynamicDataset(params=args.dataset)
    Y, T, X = hddataset.generate_observational_data(policy=None,  seed=args.dataset.get('seed', 2024))
    train_dataset, val_dataset = hddataset.get_processed_data(Y, T, X)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.exp.batch_size,
        shuffle=True,
        num_workers=args.exp.num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.exp.batch_size,
        shuffle=False,
        num_workers=args.exp.num_workers,
        drop_last=False
    )

    if args.exp.logging:
        experiment_name = args.exp.exp_name
        if args.exp.load_pretrained:
            run_id = args.exp.pretrained_path.split('/')[2]
            mlf_logger_nuisance = FilteringMlFlowLogger(
                filter_submodels=[], 
                experiment_name=experiment_name, 
                tracking_uri=args.exp.mlflow_uri, 
                run_id=run_id
            )
        else:
            mlf_logger_nuisance = FilteringMlFlowLogger(
                filter_submodels=[], 
                experiment_name=experiment_name, 
                tracking_uri=args.exp.mlflow_uri, 
                run_name=f'nuisance_run_seed{args.dataset.seed}'
            )
        artifacts_path_nuisance = hydra.utils.to_absolute_path(
            mlf_logger_nuisance.experiment.get_run(mlf_logger_nuisance.run_id).info.artifact_uri
        ).replace('mlflow-artifacts:', 'mlruns')
        logger.info(f"Artifacts path: {artifacts_path_nuisance}")
    else:
        mlf_logger_nuisance = None
        artifacts_path_nuisance = None
    
    #initialise model or load from checkpoint
    if args.exp.load_pretrained:
        original_cwd = get_original_cwd()
        pretrained_path = os.path.join(original_cwd, args.exp.pretrained_path).replace('/', '\\') #on Windows system
        nuisance_rnn = Nuisance_Network.load_from_checkpoint(checkpoint_path = pretrained_path)
        logger.info(f"Load checkpoint from {pretrained_path}")
    else:
        nuisance_rnn = Nuisance_Network(args)
    
    #define callbacks
    dml_rnn_callbacks = []
    if args.checkpoint.save:
        checkpoint_callback = ModelCheckpoint(
            dirpath = artifacts_path_nuisance,           
            filename = "nuisance-{epoch}-{val_loss:.4f}",
            monitor = args.checkpoint.monitor_nuisance,               
            mode = "min",                       
            save_top_k = args.checkpoint.top_k,
            verbose = True                      
        )
        dml_rnn_callbacks.append(checkpoint_callback)
    dml_rnn_callbacks += [LearningRateMonitor(logging_interval='epoch')]

    trainer = Trainer(
        max_epochs=args.exp.max_epochs_nuisance,
        callbacks=dml_rnn_callbacks,
        devices=1,
        accelerator=args.exp.accelerator,
        deterministic=True,
        logger = mlf_logger_nuisance,
        gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
        log_every_n_steps=args.exp.get('log_every_n_steps', 50),
    )
    if (args.exp.resume_train) or (args.exp.load_pretrained == False):
        trainer.fit(nuisance_rnn, train_loader, val_loader)

    if args.exp.use_regression_residual == False:
        logger.info("Begin residual inference using trained nuisance network")
        predictions = trainer.predict(nuisance_rnn, val_loader)
        all_res_Y_val = torch.cat([p[0] for p in predictions], dim=0)
        all_res_T_val = torch.cat([p[1] for p in predictions], dim=0)

        predictions = trainer.predict(nuisance_rnn, train_loader)
        all_res_Y_train = torch.cat([p[0] for p in predictions], dim=0)
        all_res_T_train = torch.cat([p[1] for p in predictions], dim=0)
    else:
        logger.info("DEBUG: Use residual computed from regression DML")
        assert args.dataset.sequence_length == args.dataset.n_periods
        resY_train = pickle.load(open(os.path.join(args.exp.residual_dir, 'resY_train.pkl'), 'rb'))
        resT_train = pickle.load(open(os.path.join(args.exp.residual_dir, 'resT_train.pkl'), 'rb'))
        resY_val = pickle.load(open(os.path.join(args.exp.residual_dir, 'resY_val.pkl'), 'rb'))
        resT_val = pickle.load(open(os.path.join(args.exp.residual_dir, 'resT_val.pkl'), 'rb'))
        all_res_Y_train, all_res_T_train = transform_residual_data(resY_train, resT_train, args.dataset.n_periods)
        all_res_Y_val, all_res_T_val = transform_residual_data(resY_val, resT_val, args.dataset.n_periods)

    
    val_dataset.add_residual_data(all_res_Y_val, all_res_T_val)
    train_dataset.add_residual_data(all_res_Y_train, all_res_T_train)

    train_loader = DataLoader(train_dataset, batch_size=args.exp.batch_size, shuffle=True, num_workers=args.exp.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.exp.batch_size, shuffle=False, num_workers=args.exp.num_workers, drop_last=False)

    if args.exp.plot_residual:
        plot_residual_distribution(mlf_logger_nuisance, all_res_Y_val, all_res_T_val, args)

    #Second phase of the model
    logger.info("residuals fed to parameter estimator network")
    if args.exp.logging:
        experiment_name = args.exp.exp_name
        mlf_logger_param = FilteringMlFlowLogger(
            filter_submodels=[], 
            experiment_name=experiment_name, 
            tracking_uri=args.exp.mlflow_uri, 
            run_name=f'parameter_seed{args.dataset.seed}'
        )
        artifacts_path_param = hydra.utils.to_absolute_path(
            mlf_logger_param.experiment.get_run(mlf_logger_param.run_id).info.artifact_uri
        ).replace('mlflow-artifacts:', 'mlruns')
        logger.info(f"Artifacts path: {artifacts_path_param}")
    else:
        mlf_logger_param = None
        artifacts_path_param = None
    if args.checkpoint.save:
        checkpoint_callback = ModelCheckpoint(
            dirpath = artifacts_path_param,
            filename = "param-{epoch}-{val_loss:.4f}",
            monitor = args.checkpoint.monitor_param,
            mode = "min",
            save_top_k = args.checkpoint.top_k,
            verbose = True
        )
    param_callbacks = []
    param_callbacks.append(checkpoint_callback)
    param_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    param_est = DynamicEffect_estimator(args, hddataset.true_effect)
    trainer_param = Trainer(
        max_epochs=args.exp.max_epochs_param,
        callbacks=param_callbacks,
        devices=1,
        accelerator=args.exp.accelerator,
        deterministic=True,
        logger = mlf_logger_param,
        detect_anomaly=True,
        gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
        log_every_n_steps=args.exp.get('log_every_n_steps', 50),
    )
    param_est.log_true_effect_moment_norm(val_loader, mlf_logger_param)
    trainer_param.validate(param_est, dataloaders=val_loader)
    trainer_param.fit(param_est, train_loader, val_loader)

    #logger.info(f"load best checkpoint from {checkpoint_callback.best_model_path}")
    #est_param_est = DynamicEffect_estimator.load_from_checkpoint(checkpoint_callback.best_model_path, 
    #                                                      args=args, 
    #                                                      true_effect=hddataset.true_effect)


    predictions_param = trainer_param.predict(param_est, val_loader)
    predicted_params = torch.cat([pred for pred in predictions_param], dim = 0)
    plot_param_est_distribution(mlf_logger_param, predicted_params, hddataset.true_effect, args)


    


if __name__ == "__main__":
    main()
