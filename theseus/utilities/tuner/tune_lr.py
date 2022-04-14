import os
import joblib
import optuna
import os.path as osp
from datetime import datetime
from theseus.opt import Opts
from theseus.opt import Config
from theseus.base.optimizers import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from theseus.base.augmentations import TRANSFORM_REGISTRY
from theseus.base.losses import LOSS_REGISTRY
from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.base.trainer import TRAINER_REGISTRY
from theseus.base.metrics import METRIC_REGISTRY
from theseus.base.models import MODEL_REGISTRY
from theseus.base.callbacks import CALLBACKS_REGISTRY
from theseus.utilities.getter import (get_instance)
from theseus.utilities.cuda import get_device
from theseus.base.pipeline import BasePipeline
from theseus.utilities.folder import get_new_folder_name
from theseus.utilities.loggers import LoggerObserver, FileLogger, ImageWriter
from theseus.utilities.tuner.tuner_callbacks import TuningCallbacks


class TuningPipeline(BasePipeline):
    """docstring for TuningPipeline."""

    def __init__(
        self,
        opt: Config,
        tune_opt: Config
    ):
        super(TuningPipeline, self).__init__()
        self.opt = opt
        self.tune_opt = tune_opt
    
    def init_registry(self):
        self.model_registry = MODEL_REGISTRY
        self.dataset_registry = DATASET_REGISTRY
        self.dataloader_registry = DATALOADER_REGISTRY
        self.metric_registry = METRIC_REGISTRY
        self.loss_registry = LOSS_REGISTRY
        self.optimizer_registry = OPTIM_REGISTRY
        self.scheduler_registry = SCHEDULER_REGISTRY
        self.callbacks_registry = CALLBACKS_REGISTRY
        self.trainer_registry = TRAINER_REGISTRY
        self.transform_registry = TRANSFORM_REGISTRY
        self.logger.text(
            "Overidding registry in pipeline...", LoggerObserver.INFO
        )


    def init_globals(self):
        # Main Loggers
        self.logger = LoggerObserver.getLogger("main") 

        # Global variables
        self.exp_name = self.opt['global']['exp_name']
        self.exist_ok = self.opt['global']['exist_ok']
        self.debug = False
        self.device_name = self.opt['global']['device']
        self.transform_cfg = Config.load_yaml(self.opt['global']['cfg_transform'])
        self.device = get_device(self.device_name)

        # Tuner settings
        self.study_name = self.tune_opt['global']['study_name']
        self.tune_target_key = self.tune_opt['global']['target_key']
        self.direction = self.tune_opt['global']['direction']
        self.lr_range = self.tune_opt['global']['lr_range']
        self.num_iteration_per_trials = self.tune_opt['global']['num_iteration_per_trials']
        self.time_limit_per_trials = self.tune_opt['global']['time_limit_per_trials'] 
        self.num_trials = self.tune_opt['global']['num_trials'] 

        # Experiment name
        if self.exp_name:
            self.savedir = os.path.join(self.opt['global']['save_dir'], self.exp_name, 'hyptune')
            if not self.exist_ok:
                self.savedir = get_new_folder_name(self.savedir)
        else:
            self.savedir = os.path.join(self.opt['global']['save_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'hyptune')
        os.makedirs(self.savedir, exist_ok=True)

        # Logging to files
        file_logger = FileLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(file_logger)
        self.logger.text(self.opt, level=LoggerObserver.INFO)
        self.logger.text(f"Everything will be saved to {self.savedir}", level=LoggerObserver.INFO)

    def init_optimizer(self):
        self.optimizer = get_instance(
            self.opt["optimizer"],
            registry=self.optimizer_registry,
            params=self.model.parameters(),
        )
        
    def init_pipeline(self, tuner_callbacks):
        self.init_train_dataloader()
        self.init_validation_dataloader()
        self.init_model_with_loss()
        self.init_metrics()
        self.init_optimizer()
        self.init_loading()
        self.init_scheduler()

        callbacks = [
            self.callbacks_registry.get("LoggerCallbacks")(),
            tuner_callbacks
        ]
        self.init_trainer(callbacks)

    def objective(self, trial):

        self.tuner_callbacks = TuningCallbacks(
            num_iterations=self.num_iteration_per_trials,
            time_limit=self.time_limit_per_trials
        )

        self.opt["optimizer"]['args']['lr'] = trial.suggest_float('lr', self.lr_range[0], self.lr_range[1], log=True)
        self.init_pipeline(self.tuner_callbacks)
        self.trainer.fit()

        target_params = self.tuner_callbacks.target_params
        return target_params[self.tune_target_key]

    def tune(self):
        """
        Evaluate the model
        """
        self.init_globals()
        self.init_registry()

        study = optuna.create_study(
            direction=self.direction, 
            study_name=self.study_name
        )

        study.optimize(self.objective, n_trials=self.num_trials)
        joblib.dump(study, osp.join(self.savedir, f"{self.study_name}.pkl"))

        self.logger.text(study.best_params, level=LoggerObserver.INFO)


if __name__ == "__main__":

    tune_opt = Opts().parse_args()
    opts = Config(tune_opt['global']['pipeline_cfg'])
    tune_pipeline = TuningPipeline(opts, tune_opt=tune_opt)
    tune_pipeline.tune()
    