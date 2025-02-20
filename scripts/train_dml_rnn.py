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
from src.models.dml_rnn import Nuisance_Network, DynamicEffect_estimator
from src.models.utils import evaluate_nuisance_mse, plot_residual_distribution, plot_de_est_distribution, transform_residual_data, plot_de_est_diff_distribution
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
    #instantiate dataset pipeline
    data_pipeline = instantiate(args.dataset, _recursive_=True)
    external_val_res_Y, external_val_res_T_disc, external_val_res_T_cont = list(), list(), list()
    disc_dim, cont_dim = args.dataset.n_treatments_disc, args.dataset.n_treatments_cont
    if args.exp.use_regression_residual == False:
        #k-fold cross-validation for nuisance network (Only on training set)
        for fold_idx, (train_data, val_data) in enumerate(data_pipeline.split_kfold_cv(args.exp.kfold)):
            logger.info(f"Nuisance training: Starting fold {fold_idx} out of {args.exp.kfold} folds ...")

            if args.exp.logging:
                experiment_name = args.exp.exp_name
                mlf_logger_fold = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
                    tracking_uri=args.exp.mlflow_uri, run_name=f"nuisance_{args.exp.kfold}foldcv_fold{fold_idx}")
                
                artifacts_path_fold = hydra.utils.to_absolute_path(
                    mlf_logger_fold.experiment.get_run(
                        mlf_logger_fold.run_id).info.artifact_uri).replace('mlflow-artifacts:', 'mlruns')
                logger.info(f"Artifacts path nuisance fold {fold_idx}: {artifacts_path_fold}")
            else:
                mlf_logger_fold = None
                artifacts_path_fold = None

            train_loader = DataLoader(
                train_data,batch_size=args.exp.batch_size,shuffle=False,num_workers=args.exp.num_workers,drop_last=False
            )
            val_loader = DataLoader(
                val_data,batch_size=args.exp.batch_size,shuffle=False,num_workers=args.exp.num_workers,drop_last=False
            )
            if args.exp.load_pretrained:
                assert len(args.exp.nuisance_run_ids) == args.exp.kfold
                run_id = args.exp.nuisance_run_ids[fold_idx]
                run_dir = os.path.join(hydra.utils.to_absolute_path(f"mlruns/{args.exp.exp_id}"), run_id)
                artifacts_dir = os.path.join(run_dir, 'artifacts')
                ckpts = [f for f in os.listdir(artifacts_dir) if f.endswith('ckpt')]
                if len(ckpts) == 0:
                    raise FileNotFoundError(f"No checkpoint found in {artifacts_dir}")
                checkpoint_path = os.path.join(artifacts_dir, ckpts[0])
                nuisance_rnn = Nuisance_Network.load_from_checkpoint(checkpoint_path=checkpoint_path)
                logger.info(f"Loaded checkpoint from {checkpoint_path} for fold {fold_idx}")
            else:
                nuisance_rnn = Nuisance_Network(args)
        
            #define callbacks
            dml_rnn_callbacks = []
            if args.checkpoint.save:
                checkpoint_callback = ModelCheckpoint(dirpath = artifacts_path_fold,           
                    filename = "fold" + str(fold_idx) + "-nuisance-{epoch}-{val_loss:.4f}",
                    monitor = args.checkpoint.monitor_nuisance, mode = "min", save_top_k = args.checkpoint.top_k, verbose = True                      
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
            fold_res_T_val_disc = torch.cat([p[1][:, :, :, :, :disc_dim] for p in predictions], dim=0) if disc_dim > 0 else None
            fold_res_T_val_cont = torch.cat([p[1][:, :, :, :, disc_dim:] for p in predictions], dim=0) if cont_dim > 0 else None
            logger.info(f"add fold{fold_idx} residuals to full dataset")
            data_pipeline.add_fold_residual_data(fold_idx, fold_res_Y_val, fold_res_T_val_disc, fold_res_T_val_cont)
            if args.exp.plot_residual:
                plot_residual_distribution(mlf_logger_fold, fold_res_Y_val, fold_res_T_val_cont, args)
            
            #evalute residual on the external validation set (not the internal val set from the folds)
            logger.info(f"fold{fold_idx}: Evaluate nuisance network on external validation set")
            external_val_loader = DataLoader(
                data_pipeline.val_data, batch_size=args.exp.batch_size, shuffle=False, num_workers=args.exp.num_workers, drop_last=False
            )
            predictions = trainer.predict(nuisance_rnn, external_val_loader)
            external_val_res_Y.append(torch.cat([p[0] for p in predictions], dim=0))
            external_val_res_T_disc.append(torch.cat([p[1][:, :, :, :, :disc_dim] for p in predictions], dim=0) if disc_dim > 0 else None)
            external_val_res_T_cont.append(torch.cat([p[1][:, :, :, :, disc_dim:] for p in predictions], dim=0) if cont_dim > 0 else None)

        #compute the mean of the k runs on the external val set
        logger.info("Compute mean residuals on external validation set and add to data pipeline")
        external_mean_res_Y = torch.stack(external_val_res_Y, dim=0).mean(dim=0)
        external_mean_res_T_disc = torch.stack(external_val_res_T_disc, dim=0).mean(dim=0) if external_val_res_T_disc[0] is not None else None
        external_mean_res_T_cont = torch.stack(external_val_res_T_cont, dim=0).mean(dim=0) if external_val_res_T_cont[0] is not None else None
        data_pipeline.add_full_residual_data('val', external_mean_res_Y, external_mean_res_T_disc, external_mean_res_T_cont)
        
        logger.info("All folds completed. Begin second-stage Dynamic Effect Estimator training.")

    else:
        #Need to change the code in the future
        logger.info("DEBUG: Use residual computed from regression DML")
        assert args.dataset.sequence_length == args.dataset.n_periods
        resY_full = pickle.load(open(os.path.join(args.exp.residual_dir, 'resY_full.pkl'), 'rb'))
        resT_full = pickle.load(open(os.path.join(args.exp.residual_dir, 'resT_full.pkl'), 'rb'))
        all_res_Y, all_res_T = transform_residual_data(resY_full, resT_full, args.dataset.n_periods)

    #refresh the train_loader and val_loader
    train_loader = DataLoader(data_pipeline.train_data, batch_size=args.exp.batch_size, shuffle=False, num_workers=args.exp.num_workers, drop_last=False)
    val_loader = DataLoader(data_pipeline.val_data, batch_size=args.exp.batch_size, shuffle=False, num_workers=args.exp.num_workers, drop_last=False)

    #Second phase of the model training
    if args.exp.logging:
        experiment_name = args.exp.exp_name
        mlf_logger_de = FilteringMlFlowLogger(filter_submodels=[],  experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri, run_name=f'de_est_seed{args.dataset.seed}')
        artifacts_path_de = hydra.utils.to_absolute_path(
            mlf_logger_de.experiment.get_run(mlf_logger_de.run_id).info.artifact_uri
        ).replace('mlflow-artifacts:', 'mlruns')
        logger.info(f"Artifacts path: {artifacts_path_de}")
    else:
        mlf_logger_de = None
        artifacts_path_de = None
    if args.checkpoint.save:
        checkpoint_callback = ModelCheckpoint(dirpath = artifacts_path_de, filename = "de-{epoch}-{val_loss:.4f}", monitor = args.checkpoint.monitor_de,
            mode = "min", save_top_k = args.checkpoint.top_k, verbose = True)
    de_callbacks = []
    de_callbacks.append(checkpoint_callback)
    de_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    de_est = DynamicEffect_estimator(args, data_pipeline.true_effect)
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

    test_loader = DataLoader(data_pipeline.test_data, batch_size=args.exp.batch_size, shuffle=False, num_workers=args.exp.num_workers, drop_last=False)
    predictions_de = trainer_de.predict(de_est, test_loader)
    predicted_de = torch.cat([pred for pred in predictions_de], dim = 0)

    #When individual true dynamic effect is available, log the mse / plot distribution
    if data_pipeline.gt_dynamic_effect_available:
        individual_true_effect = data_pipeline.compute_individual_true_dynamic_effects(X = data_pipeline.test_data.X_dynamic)
        if (len(args.dataset.hetero_inds) == 0) or (args.dataset.hetero_inds == None): #Only for Non-hetero estimation
            plot_de_est_distribution(mlf_logger_de, predicted_de, data_pipeline.true_effect, args)
        else:
            plot_de_est_diff_distribution(mlf_logger_de, predicted_de, individual_true_effect, args)
    
    logger.info("Evaluate individual treatment effect")
    T_intv_disc, T_base_disc = (np.ones((args.dataset.n_periods, disc_dim)), np.zeros((args.dataset.n_periods, disc_dim))) \
                                    if disc_dim > 0 else (None, None)
    T_intv_cont, T_base_cont = (np.ones((args.dataset.n_periods, cont_dim)), np.zeros((args.dataset.n_periods, cont_dim))) \
                                    if cont_dim > 0 else (None, None)
    logger.info(f"Interved treatment (discrete and continuous): \n {T_intv_disc} \n {T_intv_cont}")
    logger.info(f"Baseline treatment (discrete and continuous): \n {T_base_disc} \n {T_base_cont}")

    predicted_te = de_est.predict_treatment_effect(predicted_de, T_intv_disc, T_intv_cont, T_base_disc, T_base_cont)
    gt_te = data_pipeline.compute_treatment_effect('test', T_intv_disc, T_intv_cont, T_base_disc, T_base_cont)
    mse = ((gt_te - predicted_te) ** 2).mean()
    mlf_logger_de.log_metrics({"TE_mean":mse})
    
if __name__ == "__main__":
    main()
    #print
    print('Done!')
