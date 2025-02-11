import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from torch.utils.data import DataLoader, Subset, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.utils import FilteringMlFlowLogger
from src.data.synthetic_dataset import MarkovianHeteroDynamicDataset
from src.models.dml_rnn import Nuisance_Network, DynamicEffect_estimator
from src.models.dml import HeteroDynamicPanelDML
from src.models.utils import evaluate_nuisance_mse, plot_residual_distribution, plot_de_est_distribution, transform_residual_data
from src.utils import log_params_from_omegaconf_dict, create_loaders_with_indices
import pickle
import numpy as np
from sklearn.model_selection import KFold

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
    full_dataset = hddataset.get_full_dataset(Y, T, X)

    kf = KFold(
        n_splits=args.exp.kfold,
        shuffle=True,
        random_state=args.exp.seed
    )
    #Place holders of residuals
    N, SL, m, n_t = args.dataset.n_units, args.dataset.sequence_length, args.dataset.n_periods, args.dataset.n_treatments
    all_res_Y = np.zeros((N, SL - m + 1, m))
    all_res_T = np.zeros((N, SL - m + 1, m, m, n_t))

    if args.exp.use_regression_residual == False:
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):
            logger.info(f"Nuisance training: Starting fold {fold_idx} out of {args.exp.kfold} folds ...")

            if args.exp.logging:
                experiment_name = args.exp.exp_name
                mlf_logger_fold = FilteringMlFlowLogger(
                    filter_submodels=[],
                    experiment_name=experiment_name,
                    tracking_uri=args.exp.mlflow_uri,
                    run_name=f"nuisance_kfold_seed{args.dataset.seed}_fold{fold_idx}"
                )
                artifacts_path_fold = hydra.utils.to_absolute_path(
                    mlf_logger_fold.experiment.get_run(
                        mlf_logger_fold.run_id
                    ).info.artifact_uri
                ).replace('mlflow-artifacts:', 'mlruns')
                logger.info(f"Artifacts path nuisance fold {fold_idx}: {artifacts_path_fold}")
            else:
                mlf_logger_fold = None
                artifacts_path_fold = None

            train_dataset = Subset(full_dataset, train_idx)
            val_dataset   = Subset(full_dataset, val_idx)
            train_loader = DataLoader(
                train_dataset,batch_size=args.exp.batch_size,shuffle=False,num_workers=args.exp.num_workers,drop_last=False
            )
            val_loader = DataLoader(
                val_dataset,batch_size=args.exp.batch_size,shuffle=False,num_workers=args.exp.num_workers,drop_last=False
            )
            if args.exp.load_pretrained:
                assert len(args.exp.nuisance_run_ids) == args.exp.kfold
                run_id = args.exp.nuisance_run_ids[fold_idx]
                base_exp_dir = hydra.utils.to_absolute_path(f"mlruns/{args.exp.exp_id}")
                run_dir = os.path.join(base_exp_dir, run_id)
                artifacts_dir = os.path.join(run_dir, 'artifacts')
                ckpts = [f for f in os.listdir(artifacts_dir) if f.endswith('ckpt')]
                if len(ckpts) == 0:
                    raise FileNotFoundError(f"No checkpoint found in {artifacts_dir}")
                checkpoint_path = os.path.join(artifacts_dir, ckpts[0])
                nuisance_rnn = Nuisance_Network.load_from_checkpoint(checkpoint_path=checkpoint_path)
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            else:
                nuisance_rnn = Nuisance_Network(args)
        
            #define callbacks
            dml_rnn_callbacks = []
            if args.checkpoint.save:
                checkpoint_callback = ModelCheckpoint(
                    dirpath = artifacts_path_fold,           
                    filename = "fold" + str(fold_idx) + "-nuisance-{epoch}-{val_loss:.4f}",
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
                logger = mlf_logger_fold,
                gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
                log_every_n_steps=args.exp.get('log_every_n_steps', 50),
            )
            if (args.exp.resume_train) or (args.exp.load_pretrained == False):
                trainer.fit(nuisance_rnn, train_loader, val_loader)

            
            logger.info(f"fold{fold_idx}: Begin residual inference using trained nuisance network")
            predictions = trainer.predict(nuisance_rnn, val_loader)
            fold_res_Y_val = torch.cat([p[0] for p in predictions], dim=0)
            fold_res_T_val = torch.cat([p[1] for p in predictions], dim=0)
            if args.exp.plot_residual:
                plot_residual_distribution(mlf_logger_fold, fold_res_Y_val, fold_res_T_val, args)
            all_res_Y[val_idx] = fold_res_Y_val
            all_res_T[val_idx] = fold_res_T_val
        
        all_res_Y = torch.from_numpy(all_res_Y)
        all_res_T = torch.from_numpy(all_res_T)
        logger.info("All folds completed. Begin second-stage Dynamic Effect Estimator training.")

    else:
        logger.info("DEBUG: Use residual computed from regression DML")
        assert args.dataset.sequence_length == args.dataset.n_periods
        resY_full = pickle.load(open(os.path.join(args.exp.residual_dir, 'resY_full.pkl'), 'rb'))
        resT_full = pickle.load(open(os.path.join(args.exp.residual_dir, 'resT_full.pkl'), 'rb'))
        all_res_Y, all_res_T = transform_residual_data(resY_full, resT_full, args.dataset.n_periods)

    full_dataset.add_residual_data(all_res_Y, all_res_T)
    train_loader, val_loader, train_indices, val_indices = create_loaders_with_indices(
        full_dataset, args.dataset.train_val_split, args.exp.batch_size, 2, args.exp.seed
    )

    #Second phase of the model
    logger.info("residuals fed to Dynamic Effect estimator network")
    if args.exp.logging:
        experiment_name = args.exp.exp_name
        mlf_logger_de = FilteringMlFlowLogger(
            filter_submodels=[], 
            experiment_name=experiment_name, 
            tracking_uri=args.exp.mlflow_uri, 
            run_name=f'deeter_seed{args.dataset.seed}'
        )
        artifacts_path_de = hydra.utils.to_absolute_path(
            mlf_logger_de.experiment.get_run(mlf_logger_de.run_id).info.artifact_uri
        ).replace('mlflow-artifacts:', 'mlruns')
        logger.info(f"Artifacts path: {artifacts_path_de}")
    else:
        mlf_logger_de = None
        artifacts_path_de = None
    if args.checkpoint.save:
        checkpoint_callback = ModelCheckpoint(
            dirpath = artifacts_path_de,
            filename = "de-{epoch}-{val_loss:.4f}",
            monitor = args.checkpoint.monitor_de,
            mode = "min",
            save_top_k = args.checkpoint.top_k,
            verbose = True
        )
    de_callbacks = []
    de_callbacks.append(checkpoint_callback)
    de_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    de_est = DynamicEffect_estimator(args, hddataset.true_effect)
    trainer_de = Trainer(
        max_epochs=args.exp.max_epochs_de,
        callbacks=de_callbacks,
        devices=1,
        accelerator=args.exp.accelerator,
        deterministic=True,
        logger = mlf_logger_de,
        detect_anomaly=True,
        gradient_clip_val=args.exp.get('gradient_clip_val', 0.0),
        log_every_n_steps=args.exp.get('log_every_n_steps', 50),
    )
    de_est.log_true_effect_moment_norm(val_loader, mlf_logger_de)
    trainer_de.validate(de_est, dataloaders=val_loader)
    trainer_de.fit(de_est, train_loader, val_loader)

    #logger.info(f"load best checkpoint from {checkpoint_callback.best_model_path}")
    #est_de_est = DynamicEffect_estimator.load_from_checkpoint(checkpoint_callback.best_model_path, 
    #                                                      args=args, 
    #                                                      true_effect=hddataset.true_effect)


    predictions_de = trainer_de.predict(de_est, val_loader)
    predicted_de = torch.cat([pred for pred in predictions_de], dim = 0)

    if (len(args.dataset.hetero_inds) == 0) or (args.dataset.hetero_inds == None):
        plot_de_est_distribution(mlf_logger_de, predicted_de, hddataset.true_effect, args)
    
    logger.info("Evaluate individual treatment effect")
    T_intv = np.ones((args.dataset.n_periods, args.dataset.n_treatments))
    T_base = np.zeros((args.dataset.n_periods, args.dataset.n_treatments))
    logger.info(f"Interved treatment: \n {T_intv}")
    logger.info(f"Baseline treatment: \n {T_base}")

    predicted_te = de_est.predict_treatment_effect(predicted_de, T_intv, T_base)
    gt_te = hddataset.compute_treatment_effect(T_intv, T_base)[val_indices, :]
    mse = ((gt_te - predicted_te) ** 2).mean()
    mlf_logger_de.log_metrics({"TE_mean":mse})
    
if __name__ == "__main__":
    main()
    #print
    print('Done!')
