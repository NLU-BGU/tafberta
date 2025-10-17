from pathlib import Path


class Dirs:
    project_path = Path(__file__).parents[2]
    src = project_path / 'src'
    tafberta = src / 'tafberta'
    data = project_path / 'data'
    tokenizers = tafberta / 'tokenizers'

    raw = data / 'raw'
    processed = data / 'processed'

    # htberman
    htberman_raw = raw / 'htberman'
    htberman_raw_hebrew = htberman_raw / 'hebrew_conversations'
    htberman_raw_english = htberman_raw / 'english_conversations'
    htberman_processed = processed / 'htberman'
    htberman_processed_hebrew = htberman_processed / 'hebrew_conversations'
    htberman_processed_english = htberman_processed / 'english_conversations'

    # heclimp
    heclimp_root = processed / 'heclimp'
    heclimp_legal_words = heclimp_root / 'legal_words'
    heclimp_testsuits = heclimp_root / 'testsuits'
    heclimp_testsuits_htberman = heclimp_testsuits / 'htberman'

    # wikipedia sentences file was created using https://github.com/NLPH/SVLM-Hebrew-Wikipedia-Corpus
    wikipedia_data_raw = raw / 'wikipedia' / 'SVLM_Hebrew_Wikipedia_Corpus.txt'
    wikipedia_corpus_processed = processed / 'htberman' / 'wikipedia' / 'wikipedia_segmented.txt'
    wikipedia_testsuits = heclimp_root / 'wikipedia'

    # mlflow
    mlflow_tracking_uri = project_path / 'mlruns'


class DataPrep:
    all_datasets = ['BSF',
                    'BatEl',
                    'BermanLong',
                    'Levy',
                    'Ravid']
    childes_in_htberman_dataset = 'BermanLong'
    
    url = "https://childes.talkbank.org/data/Other/Hebrew/%s.zip"


class Data:
    htberman_processed_corpus = Dirs.htberman_processed / 'corpus.txt'

    min_sentence_length = 2
    train_prob = 1.0  # probability that sentence is assigned to train split
    mask_symbol = '<mask>'
    pad_symbol = '<pad>'
    unk_symbol = '<unk>'
    bos_symbol = '<s>'
    eos_symbol = '</s>'
    roberta_symbols = [mask_symbol, pad_symbol, unk_symbol, bos_symbol, eos_symbol]
    

class Tokenizer:
    vocab_size = 2 * 4096  # as the vocab size of BabyBERTa
    min_frequency = 2
    add_prefix_space = True  # as in AlephBert and BabyBERTa
    corpus_file_paths = [str(Dirs.htberman_processed)]  # tokenizer.train(files=...) expects list of str paths
    tokenizer_path = Dirs.tokenizers / 'htberman_tokenizer.json'


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
  
    paradigm_paths = [
        Dirs.heclimp_testsuits / 'agreement_determiner_noun-across_0_adjective_num',
        Dirs.heclimp_testsuits / 'agreement_determiner_noun-across_0_adjective_gen'
        ]

    # paradigm_paths = [
    #     Dirs.wikipedia_testsuits / 'agreement_determiner_noun-across_0_adjective_num',
    #     Dirs.wikipedia_testsuits / 'agreement_determiner_noun-across_0_adjective_gen'
    #     ]


class Scoring:
    batch_size = 10
