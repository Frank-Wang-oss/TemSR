import sys

import os

# path problem.
import torch.cuda

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import pandas as pd
import collections
import argparse
import warnings
import sklearn.exceptions
from configs.sweep_params import get_sweep_hparams

from utils import fix_randomness, starting_logs, AverageMeter
from abstract_trainer import AbstractTrainer
import wandb

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)



    def sweep(self):
        # sweep configurations

        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_maximize, 'goal': 'maximize'},
            'name': self.da_method + '_' + self.backbone,
            'parameters': {**get_sweep_hparams(self.dataset)[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)  # Training with sweep

    def train(self):

        if self.is_sweep:
            run = wandb.init(config=self.default_hparams)
            run_name = f"sweep_{self.dataset}"
        else:
            run_name = f"{self.run_description}"
            run = wandb.init(config=self.default_hparams, mode="online", name=run_name)

        self.hparams = wandb.config
        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id)
                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id) # src_train: [1676, 110, 128], trg_train: [1555, 110, 128]

                # Train model
                non_adapted_model, last_adapted_model, best_adapted_model = self.train_model()

                # Save checkpoint ## Uncomment the below code if you need to save checkpoint
                # self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model, last_adapted_model, best_adapted_model)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"

                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)
        avg_acc = table_results[table_results['scenario'] == 'mean']['acc'].values[0]
        avg_f1 = table_results[table_results['scenario'] == 'mean']['f1_score'].values[0]
        avg_src_risk = table_risks[table_risks['scenario'] == 'mean']['src_risk'].values[0]
        avg_trg_risk = table_risks[table_risks['scenario'] == 'mean']['trg_risk'].values[0]

        average_metrics = {'acc': avg_acc, 'f1_score': avg_f1, 'src_risk': avg_src_risk, 'trg_risk': avg_trg_risk}
        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

        wandb.log(average_metrics)
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log({'avg_results': wandb.Table(dataframe=table_results, allow_mixed_types=True)})
        run.finish()

if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs', type=str,
                        help='Directory containing all experiments')
    parser.add_argument('--run_description', default='train', type=str, help='Description of run, if none, DA method name will be used')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='TemSR', type=str, help='NRC, AaD, SHOT, MAPU,')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'../Datasets', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (EEG - HAR - FD)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone: CNN')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda:0')
    parser.add_argument('--gpu_id', default=0, type=str, help='gpu id.')

    # ========= Sweep settings ===============
    parser.add_argument('--num_sweeps', default=30, type=str, help='Number of sweep runs')
    parser.add_argument('--sweep_project_wandb', default='random_Testing_v7', type=str, help='Project name in Wandb')
    parser.add_argument('--wandb_entity', type=str,
                        help='Entity name in Wandb (can be left blank if there is a default entity)')
    parser.add_argument('--hp_search_strategy', default="random", type=str,
                        help='The way of selecting hyper-parameters (random-grid-bayes). in wandb see:https://docs.wandb.ai/guides/sweeps/configuration')
    parser.add_argument('--metric_to_maximize', default="f1_score", type=str)
    parser.add_argument('--is_sweep', default=False, type=bool, help='singe run or sweep')


    args = parser.parse_args()

    trainer = Trainer(args)
    if args.is_sweep:
        trainer.sweep()
        print(1)
    else:
        trainer.train()
        print(2)
