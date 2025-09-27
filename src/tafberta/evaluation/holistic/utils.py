from typing import List

import pandas as pd
import numpy as np


def get_legend_label(model_name):
    return model_name


def get_correct_vector_results_from_scores(scores: List[float],
                                           scoring_method: str='holistic'
                                           ) -> np.array:
    """
    Get a np.array of correct model predictions.

    Notes:
        Odd-numbered lines contain grammatical sentences, an even-numbered lines contain ungrammatical sentences.
        
        Whether a pair is scored as correct depends on the scoring method:
        - When scoring with MLM method, the pseudo-log-likelihood must be larger for the correct sentence.
        - When scoring with the holistic method, the cross-entropy error sum must be smaller for the correct sentence.
    """

    scores = np.array(scores, dtype=float)  # Convert the scores list to a NumPy array for vectorized operations
    num_pairs = len(scores) // 2

    # Split the scores into grammatical and ungrammatical pairs
    grammatical_scores = scores[::2]  # Odd-indexed scores (grammatical sentences)
    ungrammatical_scores = scores[1::2]  # Even-indexed scores (ungrammatical sentences)

    if scoring_method == 'mlm':
        correct = grammatical_scores < ungrammatical_scores  # Grammatical sentences should have smaller scores
    elif scoring_method == 'holistic':
        correct = grammatical_scores > ungrammatical_scores  # Grammatical sentences should have larger scores
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    return correct


def calc_accuracy_from_scores(scores: List[float],
                              scoring_method: str='holistic'
                             ) -> float:
    """
    Compute accuracy given scores.

    Notes:
        Odd-numbered lines contain grammatical sentences, an even-numbered lines contain ungrammatical sentences.
        
        Whether a pair is scored as correct depends on the scoring method:
        - When scoring with MLM method, the pseudo-log-likelihood must be larger for the correct sentence.
        - When scoring with the holistic method, the cross-entropy error sum must be smaller for the correct sentence.
    """

    scores = np.array(scores, dtype=float)  # Convert the scores list to a NumPy array for vectorized operations
    num_pairs = len(scores) // 2

    # Split the scores into grammatical and ungrammatical pairs
    grammatical_scores = scores[::2]  # Odd-indexed scores (grammatical sentences)
    ungrammatical_scores = scores[1::2]  # Even-indexed scores (ungrammatical sentences)

    if scoring_method == 'mlm':
        correct = grammatical_scores < ungrammatical_scores  # Grammatical sentences should have smaller scores
    elif scoring_method == 'holistic':
        correct = grammatical_scores > ungrammatical_scores  # Grammatical sentences should have larger scores
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    # Calculate accuracy as the percentage of correctly scored pairs
    accuracy = np.mean(correct) * 100  # `correct` is a boolean array, `mean()` gives the percentage of True values

    return accuracy

def get_group_names(df: pd.DataFrame,
                    ) -> List[str]:
    df['group_name'] = df['model'].str.cat(df['corpora'], sep='+').astype('category')
    res = df['group_name'].unique().tolist()
    return res