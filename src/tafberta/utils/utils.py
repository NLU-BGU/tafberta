import random
import torch
from torch.nn import CrossEntropyLoss
from typing import Tuple, List, Dict
from itertools import islice
from transformers import PreTrainedTokenizerFast
from transformers.models.roberta import RobertaForMaskedLM
from transformers import RobertaTokenizer
import re
import mlflow.pytorch

import pylangacq

from tafberta import configs

loss_fct = CrossEntropyLoss()

# Functions for data preparation

def get_children(file_paths: List[str]) -> List[str]:
    """
    Suit for BatEl, BermanLong and Levy datasets
    """
    return list(set([path.split('/')[1] for path in file_paths]))

def get_not_child_participants(dataset: pylangacq.chat.Reader) -> List[str]:
    return list(set([parti for header in dataset.headers()
                    for parti, meta in header['Participants'].items()
                    if meta['role'] != 'Target_Child']))

def get_child_participants(dataset: pylangacq.chat.Reader) -> List[str]:
    return list(set([parti for header in dataset.headers()
                    for parti, meta in header['Participants'].items()
                    if meta['role'] == 'Target_Child']))

def is_dataset_has_morphology(dataset: pylangacq.chat.Reader):
    utt = dataset.utterances(participants=get_not_child_participants(dataset))
    return '%mor' in(utt[0].tiers.keys())

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def del_list_by_ids(l, id_to_del):
    if id_to_del == []:
        return l
    return [i for j, i in enumerate(l) if j not in id_to_del]

def get_eng_cleaned_sentences(reader_eng: pylangacq.chat.Reader,
                              extra_lines_ids: List
                             ) -> List[str]:
    eng_utt = reader_eng.utterances()
        
    # in all the files, the extra lines are in eng files
    eng_utt_cleaned = del_list_by_ids(eng_utt, extra_lines_ids)
    eng_sentences = [' '.join([token.word for token in utt.tokens])
                     for utt in eng_utt_cleaned]
    return eng_sentences, eng_utt_cleaned


# Functions for training and evaluation

def make_sequences(sentences: List[str],
                   num_sentences_per_input: int,
                   ) -> List[str]:

    gen = (bs for bs in sentences)

    # combine multiple sentences into 1 sequence
    res = []
    while True:
        sentences_in_sequence: List[str] = list(islice(gen, 0, num_sentences_per_input))
        if not sentences_in_sequence:
            break
        sequence = ' '.join(sentences_in_sequence)
        res.append(sequence)

    print(f'Num total sequences={len(res):,}', flush=True)
    return res


def split(data: List[str],
          seed: int = 2) -> Tuple[List[str],
                                  List[str]]:

    print(f'Splitting data into train/dev sets...')

    random.seed(seed)

    train = []
    devel = []

    for i in data:
        if random.choices([True, False],
                          weights=[configs.Data.train_prob, 1 - configs.Data.train_prob])[0]:
            train.append(i)
        else:
            devel.append(i)

    print(f'num train sequences={len(train):,}', flush=True)
    print(f'num devel sequences={len(devel):,}', flush=True)

    return train, devel


def forward_mlm(model,
                mask_matrix: torch.bool,  # mask_matrix is 2D bool array specifying which tokens to predict
                x: Dict[str, torch.tensor],
                y: torch.tensor,
                ) -> torch.tensor:
    output = model(**{k: v.to('cuda') for k, v in x.items()})
    logits_3d = output['logits']
    logits_2d = logits_3d.view(-1, model.config.vocab_size)
    bool_1d = mask_matrix.view(-1)
    logits_for_masked_words = logits_2d[bool_1d]
    labels = y.view(-1).cuda()
    loss = loss_fct(logits_for_masked_words,  # [num masks in batch, vocab size]
                    labels)  # [num masks in batch]

    return loss


def predict_masked_token(masked_seq: str,
                         tokenizer: PreTrainedTokenizerFast,
                         model: RobertaForMaskedLM):
    seq_encoding = tokenizer(masked_seq, add_special_tokens=True, return_tensors="pt")
    output = model(**seq_encoding)
    logits = output.logits
    
    # plus 1 because we add a special token at the beginning of the seq
    mask_index = [i for i, x in enumerate(masked_seq.split())
                  if x == configs.Data.mask_symbol][0] + 1
    predicted_token_id = logits[0, mask_index].argmax(axis=-1)
    return tokenizer.decode(predicted_token_id)


def get_top_k_mask(masked_seq: str,
                   k: int,
                   tokenizer: PreTrainedTokenizerFast,
                   model: RobertaForMaskedLM
                  ) -> Tuple[str, torch.Tensor]:
    seq_encoding = tokenizer(masked_seq, add_special_tokens=True, return_tensors="pt")
    output = model(**seq_encoding)
    logits = output.logits
    
    # plus 1 because we add a special token at the beginning of the seq
    mask_index = [i for i, x in enumerate(masked_seq.split())
                  if x == configs.Data.mask_symbol][0] + 1
    predicted_token_ids = torch.topk(logits[0, mask_index], k, dim=-1).indices
    predicted_token_logits = torch.topk(logits[0, mask_index], k, dim=-1).values.detach()
    return tokenizer.decode(predicted_token_ids).split(), predicted_token_logits

def count_unique_words(sentences: List[str]) -> Tuple[int, set]:
    """
    Counts the number of unique words in a list of sentences.

    Args:
        sentences (List[str]): A list of sentences represented as strings.

    Returns:
        Tuple[int, set]: 
            - An integer representing the count of unique words.
            - A set containing the unique words found in the input sentences.

    Notes:
        - Words are defined as sequences of alphabetic characters in the range [א-ת].
        - The comparison is case-insensitive.
    """
    # Combine all sentences into a single string
    combined_text = ' '.join(sentences)
    
    # Use regular expression to find all words consisting of alphabetic characters only
    words = re.findall(r'\b[א-ת]+\b', combined_text.lower())
    
    # Create a set to store unique words
    unique_words = set(words)
    
    # Return the number of unique words
    return len(unique_words), unique_words

def count_words(sentences: List[str]) -> Tuple[int, List[str]]:
    """
    Counts the total number of words in a list of sentences.

    Args:
        sentences (List[str]): A list of sentences represented as strings.

    Returns:
        Tuple[int, List[str]]: 
            - An integer representing the total count of words.
            - A list containing all the words found in the input sentences.

    Notes:
        - Words are defined as sequences of alphabetic characters in the range [א-ת].
        - The comparison is case-insensitive.
    """
    # Combine all sentences into a single string
    combined_text = ' '.join(sentences)
    
    # Use regular expression to find all words consisting of alphabetic characters only
    words = re.findall(r'\b[א-ת]+\b', combined_text.lower())
    
    # Return the number of unique words
    return len(words), words

def load_model(run_id, model_dir='best_model'):
    """
    Load a PyTorch model from MLflow using a specified run ID and artifact path.

    Args:
        run_id (str): The ID of the MLflow run where the model was logged.
        model_dir (str, optional): The artifact path where the model is stored within the run. 
                                   Defaults to 'best_model'.

    Returns:
        torch.nn.Module: The loaded PyTorch model.

    Notes:
        - Constructs the model URI based on the provided run ID and model directory.
        - Loads the model using `mlflow.pytorch.load_model`.
    """
    model_uri = f"runs:/{run_id}/{model_dir}"
    model = mlflow.pytorch.load_model(model_uri)
    return model


# Load all models logged in MLflow for an experiment
def load_models_from_mlflow(experiment_name):
    """
    Load all finished models from an MLflow experiment by its name.

    Args:
        experiment_name (str): The name of the MLflow experiment.

    Returns:
        List[Tuple[str, torch.nn.Module]]: A list of (run_id, model) tuples,
        where 'run_id' is the ID of the run and 'model' is the loaded PyTorch model.

    Raises:
        ValueError: If the specified experiment does not exist.

    Details:
        - Only models from runs with status 'FINISHED' are loaded.
        - Assumes each model is logged under the artifact path 'best_model'.
        - Only active (non-deleted) runs are considered.
    """
    models = []
    client = mlflow.tracking.MlflowClient()
    
    # Fetch the experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")
    experiment_id = experiment.experiment_id
    
    # List all runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )
    
    # Load each model from its logged location
    for run in runs:
        model_uri = f"runs:/{run.info.run_id}/best_model"
        model = mlflow.pytorch.load_model(model_uri)
        models.append((run.info.run_id, model))
    
    return models