# mlflow ui --port 9990 --host 0.0.0.0

import torch
import random
from typing import Dict, Union
import optuna
import mlflow
import tempfile
import os
import statistics
from pprint import pformat
import inspect
import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM, RobertaConfig, get_linear_schedule_with_warmup, AdamW

from tafberta import configs
from tafberta.utils.io import load_sentences_from_file, load_tokenizer, load_PreTrainedTokenizerFast
from tafberta.params import Params, param2default
from tafberta.utils.utils import split, make_sequences
from tafberta.dataset import DataSet
from tafberta.evaluation.scoring_minimal_pairs import ModelScorer
from tafberta.training.dataloading import collate_fn

# Set up or fetch experiment by name, and get its ID
experiment_name = configs.Training.experiment_name
mlflow.set_tracking_uri(configs.Dirs.mlflow_tracking_uri)  # Local storage (can be changed to remote URI if needed)
experiment = mlflow.get_experiment_by_name(experiment_name)  # if experiment exists - get from mlruns db

# Create experiment if it doesn't exist
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id


# Fetch all previous runs and their parameters
def get_previous_combinations(experiment_id):
    # Search for all completed runs in the current experiment
    previous_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="status = 'FINISHED'")
    
    # Extract parameters from the runs into a set of tuples
    combinations = set()
    
    for index, row in previous_runs.iterrows():
        param_tuple = (
            row.get('params.num_attention_heads'),
            row.get('params.hidden_size'),
            row.get('params.leave_unmasked_prob'),
            row.get('params.num_layers'),
            row.get('params.intermediate_size'),
        )
        combinations.add(param_tuple)
    
    return combinations


def _load_tokenizer(tokenizer_path, max_input_length):
    tokenizer = load_tokenizer(tokenizer_path, max_input_length)
    return tokenizer


def transform_string_log(input_str):
    parts = input_str.split('_')
    result = f"{parts[0]}_{parts[-1]}"
    return result


class TafBERTaLightningModule(pl.LightningModule):
    def __init__(self, params):
        super(TafBERTaLightningModule, self).__init__()
        self.params = params
        self.paradigm_paths = configs.Eval.paradigm_paths  # List of paradigm file paths
        self.tokenizer_path = configs.Dirs.tokenizers / f'{params.tokenizer}.json'
        self.scorer = ModelScorer(model_path=None, tokenizer_path=self.tokenizer_path)
        self.save_hyperparameters()

        # Model configuration
        self.tokenizer = _load_tokenizer(self.tokenizer_path, self.params.max_input_length)
        self.tokenizerFast = load_PreTrainedTokenizerFast(self.tokenizer_path, self.params.max_input_length)
        vocab_size = len(self.tokenizer.get_vocab())
        print(f'Vocab size={vocab_size}')
        
        self.config = RobertaConfig(
            vocab_size=vocab_size,
            pad_token_id=self.tokenizer.token_to_id(configs.Data.pad_symbol),
            bos_token_id=self.tokenizer.token_to_id(configs.Data.bos_symbol),
            eos_token_id=self.tokenizer.token_to_id(configs.Data.eos_symbol),
            return_dict=True,
            is_decoder=False,
            is_encoder_decoder=False,
            add_cross_attention=False,
            layer_norm_eps=params.layer_norm_eps,
            max_position_embeddings=params.max_input_length + 2,
            hidden_size=params.hidden_size,
            num_hidden_layers=params.num_layers,
            num_attention_heads=params.num_attention_heads,
            intermediate_size=params.intermediate_size,
            initializer_range=params.initializer_range,
        )
        
        # initialize model
        self.model = RobertaForMaskedLM(config=self.config)
        
        self.loss = None

        # Initialize variables to track the best accuracy_dev and corresponding test accuracy and epoch
        self.best_accuracy_dev = -float('inf')  # Initialize to negative infinity
        self.best_accuracy_test = None
        self.best_epoch = None
        self.best_model_state_dict = None
        self.accuracy_test_per_epoch = {}  # Dictionary to store accuracy_test per epoch

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        # print(f'batch_idx: {batch_idx}')
        x_mm, y = batch
        x, mm = x_mm['input_ids'], x_mm['attention_mask']
        
        outputs = self(x, mm, y)
        loss = outputs.loss
        mlflow.log_metric('train_loss', loss, step=batch_idx)
        return loss

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        mlflow.log_metric(f"epoch", epoch)

        test_paradigm_paths = [pp / 'test.txt'
                                for pp in self.paradigm_paths]
        dev_paradigm_paths = [pp / 'dev.txt'
                              for pp in self.paradigm_paths]
        
        accuracy_scores_test = self.scorer.compute_scores_with_model(model=self.model,
                                                                     paradigm_paths=test_paradigm_paths,
                                                                     tokenizer=self.tokenizerFast,
                                                                     device=self.device
                                                                     )
        accuracy_scores_dev = self.scorer.compute_scores_with_model(model=self.model,
                                                                    paradigm_paths=dev_paradigm_paths,
                                                                    tokenizer=self.tokenizerFast,
                                                                    device=self.device
                                                                   )
        
        for paradigm_name, score in accuracy_scores_test.items():
            paradigm_name_log = transform_string_log(paradigm_name)
            mlflow.log_metric(f"test_accuracy_{paradigm_name_log}", score, step=epoch)
            print(f"Logged accuracy for test_{paradigm_name_log}: {score}")
            
        for paradigm_name, score in accuracy_scores_dev.items():
            paradigm_name_log = transform_string_log(paradigm_name)
            mlflow.log_metric(f"dev_accuracy_{paradigm_name_log}", score, step=epoch)
            print(f"Logged accuracy for dev_{paradigm_name_log}: {score}")
        
        average_all = statistics.mean(list(accuracy_scores_test.values())
                                      + list(accuracy_scores_dev.values()))
        mlflow.log_metric(f"accuracy_combined", average_all, step=epoch)
        print(f"Logged accuracy_combined: {average_all}")
        
        average_test = statistics.mean(list(accuracy_scores_test.values()))
        mlflow.log_metric(f"accuracy_test", average_test, step=epoch)
        print(f"Logged accuracy_test: {average_test}")
        
        average_dev = statistics.mean(list(accuracy_scores_dev.values()))
        mlflow.log_metric(f"accuracy_dev", average_dev, step=epoch)
        self.log("accuracy_dev", average_dev, prog_bar=True)  # Log to PL for early stopping and checkpointing
        print(f"Logged accuracy_dev: {average_dev}")

        # Store accuracy_test per epoch
        self.accuracy_test_per_epoch[epoch] = average_test

        # Update best accuracy_dev and corresponding test accuracy and epoch
        if average_dev > self.best_accuracy_dev:
            self.best_accuracy_dev = average_dev
            self.best_accuracy_test = average_test
            self.best_epoch = epoch
            # Save the model's state dict
            self.best_model_state_dict = self.model.state_dict()

    def on_train_end(self):
        # Log the final model at the end of training to MLflow
        mlflow.pytorch.log_model(self.model, "models")

        # Log the accuracy_dev_max and accuracy_test_final to MLflow
        mlflow.log_metric("accuracy_dev_max", self.best_accuracy_dev)
        mlflow.log_metric("accuracy_test_final", self.best_accuracy_test)
        mlflow.log_metric("epoch_max", self.best_epoch)
        

        # Log the best model according to accuracy_dev_max
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save the best model to a temporary directory
            model_save_path = os.path.join(tmp_dir, "best_model")
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(self.best_model_state_dict, os.path.join(model_save_path, "pytorch_model.bin"))
            # Load the model for logging
            best_model = RobertaForMaskedLM(config=self.config)
            best_model.load_state_dict(self.best_model_state_dict)
            # Log the best model to MLflow
            mlflow.pytorch.log_model(best_model, "best_model")
            print("Logged the best model to MLflow.")

    def validation_step(self, batch, batch_idx):
        x_mm, y = batch
        x, mm = x_mm['input_ids'], x_mm['attention_mask']
        
        outputs = self(x, mm, y)
        loss = outputs.loss
        mlflow.log_metric('val_loss', loss, step=batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
            correct_bias=False,
            no_deprecation_warning=True  # Ignore deprecation warnings
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.params.num_warmup_steps,
            num_training_steps=self.trainer.max_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch' for step per epoch
                'frequency': 1       # Step every 'n' steps/epochs
            }
        }


class TafBERTaDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super(TafBERTaDataModule, self).__init__()
        self.params = params
        self.tokenizer_path = configs.Dirs.tokenizers / f'{params.tokenizer}.json'
        self.tokenizer = _load_tokenizer(self.tokenizer_path, self.params.max_input_length)
        
    def prepare_data(self):
        # Download or prepare data if needed
        pass

    def setup(self, stage=None):
        project_path = configs.Dirs.project_path
        sentences = []

        for corpus_name in self.params.corpora:
            if corpus_name.lower() in {"htberman", "cds"}:
                data_path = configs.Data.htberman_processed_corpus
                print(f'Using corpus: {data_path}')
            sentences_in_corpus = load_sentences_from_file(
                data_path,
                include_punctuation=self.params.include_punctuation,
                allow_discard=True
            )
            sentences += sentences_in_corpus

        if self.params.training_order == 'shuffled':
            random.shuffle(sentences)
        elif self.params.training_order == 'reversed':
            sentences = sentences[::-1]
        elif self.params.training_order == 'original':
            pass
        else:
            raise AttributeError('Invalid arg to training_order.')

        all_sequences = make_sequences(sentences, self.params.num_sentences_per_input)
        self.all_sequences = all_sequences
        train_sequences, devel_sequences = split(all_sequences)

        # train_sequences = train_sequences[:10]
        
        self.train_dataset = DataSet(train_sequences, self.tokenizer, self.params)
        self.val_dataset = DataSet(devel_sequences, self.tokenizer, self.params)
        print(f'len train data: {len(self.train_dataset)}')
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.params.batch_size,
                          shuffle=False,
                          collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
                          num_workers=5,
                         )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.params.batch_size,
                          shuffle=False,
                          collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
                          num_workers=5,
                         )


def get_immediate_caller_func_name():
    # Get the current call stack
    stack = inspect.stack()
    
    # The immediate caller is the second item in the stack (index 1)
    caller_frame = stack[2]  # stack[0] is the current function's frame

    # Get the name of the function that called this function
    caller_function_name = caller_frame.function
    return caller_function_name


# train on specified params
def main(
    param2val: Union[Params, Dict],
    patience=None,
    ):
    if type(param2val) != Params:
        params = Params.from_param2val(param2val)
    else:
        params = param2val
    params.framework = 'huggingface'
    params.is_huggingface_recommended = False

    if not patience:
        patience = configs.Training.patience    

    model = TafBERTaLightningModule(params)
    data_module = TafBERTaDataModule(params)
    
    # Set up early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=configs.Training.monitor,  # The metric to monitor
        patience=patience,                 # Number of epochs to wait for improvement
        verbose=True,                      # Prints a message when early stopping is triggered
        mode=configs.Training.mode
    )
    
    # Set up model checkpoint callback to save the best model according to accuracy_dev
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor='accuracy_dev',  # The metric name to monitor
        filename='best_model_epoch{epoch:02d}',
        save_top_k=1,
        mode='max',  # or 'min' depending on the metric
    )
    
    trainer = pl.Trainer(
        max_epochs=params.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=1,
        # devices=4 if torch.cuda.is_available() else 1,
        logger=False, #mlflow_logger,
        enable_checkpointing=True,
        # precision=16,
        # accumulate_grad_batches=4,
        # strategy='deepspeed_stage_2',  # Stage 2 is recommended for balanced memory saving
    )
    
    immediate_caller_func_name = get_immediate_caller_func_name()
    if immediate_caller_func_name != 'objective':
        with mlflow.start_run(experiment_id=experiment_id,
                          # run_name=f"trial_{trial.number}",
                          nested=False,
                         ):
            print(f'The main func called from func named: {immediate_caller_func_name}')
            mlflow.log_param('num_attention_heads', params.num_attention_heads)
            mlflow.log_param('hidden_size', params.hidden_size)
            mlflow.log_param('leave_unmasked_prob', params.leave_unmasked_prob)
            mlflow.log_param('num_layers', params.num_layers)
            mlflow.log_param('intermediate_size', params.intermediate_size)
            mlflow.log_param('batch_size', params.batch_size)
            mlflow.log_param('patience', patience)
            print('Params logged to mlflow')
    
            trainer.fit(model, data_module)
    
            # Log the monitored metric manually to MLflow
            val_metric = trainer.callback_metrics[configs.Training.monitor].item()
            mlflow.log_metric(configs.Training.monitor, val_metric)

            # After training, log the best model according to accuracy_dev_max
            # The best model is saved by the checkpoint_callback
            best_model_path = checkpoint_callback.best_model_path
            if best_model_path:
                # Load the best model
                best_model = TafBERTaLightningModule.load_from_checkpoint(best_model_path, params=params)
                # Log the best model to MLflow
                mlflow.pytorch.log_model(best_model.model, "best_model")
                print("Logged the best model to MLflow.")

                # Extract the epoch number from the filename
                match = re.search(r'epoch(\d+)', best_model_path)
                if match:
                    best_epoch = int(match.group(1))
                    # Get the accuracy_test at the best epoch
                    accuracy_test_final = model.accuracy_test_per_epoch.get(best_epoch, None)
                    if accuracy_test_final is not None:
                        mlflow.log_metric('accuracy_test_final', accuracy_test_final)
                    else:
                        print("Accuracy test final value not found.")
                else:
                    print("Could not extract epoch from best_model_path.")
            else:
                print("Best model path not found.")

            # Log the 'accuracy_dev_max' metric
            accuracy_dev_max = checkpoint_callback.best_model_score.item()
            mlflow.log_metric('accuracy_dev_max', accuracy_dev_max)
    
    else:
        trainer.fit(model, data_module)
    
    return trainer

# Obtain hyperparameters for this trial
def suggest_hyperparameters(trial):
    num_attention_heads = trial.suggest_categorical('num_attention_heads', list(range(2, 13, 2)))
    hidden_size = trial.suggest_categorical('hidden_size', [2 ** i for i in range(6, 10)] + [768])
    leave_unmasked_prob = trial.suggest_categorical('leave_unmasked_prob', [0.0, 0.1])
    num_layers = trial.suggest_categorical('num_layers', list(range(2, 13, 2)))
    intermediate_size = trial.suggest_categorical('intermediate_size', [2 ** i for i in range(6, 13)] + [3072])

    print(f"Suggested hyperparameters: \n{pformat(trial.params)}")
    return num_attention_heads, hidden_size, leave_unmasked_prob, num_layers, intermediate_size


# hyperparameters finetuning using optuna
def objective(trial):
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~ trial.number: {trial.number} ~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    # Get the previous combinations tried in MLflow
    previous_combinations = get_previous_combinations(experiment_id)
    
    # Hyperparameter tuning
    num_attention_heads, hidden_size, leave_unmasked_prob, num_layers, intermediate_size = suggest_hyperparameters(trial)

    # Check if the hidden size is divisible by the number of attention heads
    if hidden_size % num_attention_heads != 0:
        # Prune the trial if the constraint is not satisfied
        raise optuna.TrialPruned(f"Invalid combination: hidden_size ({hidden_size}) is not divisible by num_attention_heads ({num_attention_heads})")

    # Create the current combination tuple
    current_combination = (num_attention_heads, hidden_size, leave_unmasked_prob, num_layers, intermediate_size)
    
    # Check if the current combination has already been tried
    if current_combination in previous_combinations:
        print(f"Combination {current_combination} already tried. Pruning this trial.")
        raise optuna.TrialPruned("Combination already tried in previous runs")
        
    with mlflow.start_run(experiment_id=experiment_id,
                          nested=False,
                         ):
        
        # mlflow.log_params(trial.params)
        mlflow.log_param('num_attention_heads', num_attention_heads)
        mlflow.log_param('hidden_size', hidden_size)
        mlflow.log_param('leave_unmasked_prob', leave_unmasked_prob)
        mlflow.log_param('num_layers', num_layers)
        mlflow.log_param('intermediate_size', intermediate_size)

        patience = configs.Training.patience   
        mlflow.log_param('patience', patience)
        
        # Set up parameters for model and data module
        param2val = param2default  #assume param2default is imported
        params = Params.from_param2val(param2val)
        params.leave_unmasked_prob = leave_unmasked_prob
        params.leave_unmasked_prob_start = leave_unmasked_prob
        params.num_layers = num_layers
        params.num_attention_heads = num_attention_heads
        params.hidden_size = hidden_size
        params.intermediate_size = intermediate_size
        
        mlflow.log_param('batch_size', params.batch_size)
        
        trainer = main(params)
        
        # Log the monitored metric manually to MLflow
        val_metric = trainer.callback_metrics[configs.Training.monitor].item()
        mlflow.log_metric(configs.Training.monitor, val_metric)
    
        return val_metric
