import optuna
from tafberta import configs
from tafberta.training.train_lightning import objective

experiment_name = configs.Training.experiment_name

# Create a study and optimize the objective
study = optuna.create_study(
    direction='maximize',
    study_name=experiment_name,
)

study.optimize(objective, n_trials=configs.Training.n_trials)

# Best hyperparameters
print("Best trial:", study.best_trial.params)