from pathlib import Path


class Dirs:
    project_path = Path(__file__).parents[2]
    src = project_path / 'src'
    tafberta = src / 'tafberta'
    data = project_path / 'data'
    tokenizers = tafberta / 'tokenizers'

    raw = data / 'raw'
    processed = data / 'processed'
    htberman_processed = processed / 'htberman' / 'corpus.txt'
    heclimp_root = processed / 'heclimp'
    heclimp_testsuits = heclimp_root / 'htberman'

    # wikipedia sentences file was created using https://github.com/akb89/witokit
    wikipedia_data_raw = raw / 'wikipedia' / 'SVLM_Hebrew_Wikipedia_Corpus.txt'
    wikipedia_corpus_processed = processed / 'htberman' / 'wikipedia' / 'wikipedia_segmented.txt'
    wikipedia_testsuits = heclimp_root / 'wikipedia'

    # mlflow
    mlflow_tracking_uri = project_path / 'mlruns'


class Data:
    min_sentence_length = 2
    train_prob = 1.0  # probability that sentence is assigned to train split
    mask_symbol = '<mask>'
    pad_symbol = '<pad>'
    unk_symbol = '<unk>'
    bos_symbol = '<s>'
    eos_symbol = '</s>'
    roberta_symbols = [mask_symbol, pad_symbol, unk_symbol, bos_symbol, eos_symbol]
    


class Training:
    feedback_interval = 1000
    max_step = 400000
    
    # optuna
    experiment_name = 'Hebrew Wikipedia'
    patience = 1  # Number of epochs to wait for improvement
    # patience = 100  # Number of epochs to wait for improvement
    n_trials = 100
    monitor = 'accuracy_dev'  # The metric to monitor for early stopping and hyperparameters funetuning
    mode = 'max'  # related to the metric to monitor


class Eval:
    interval = 20_000
  
    # paradigm_paths = [
    #     Dirs.heclimp_testsuits / 'agreement_determiner_noun-across_0_adjective_num',
    #     Dirs.heclimp_testsuits / 'agreement_determiner_noun-across_0_adjective_gen'
    #     ]

    paradigm_paths = [
        Dirs.wikipedia_testsuits / 'agreement_determiner_noun-across_0_adjective_num',
        Dirs.wikipedia_testsuits / 'agreement_determiner_noun-across_0_adjective_gen'
        ]


class Scoring:
    batch_size = 10
