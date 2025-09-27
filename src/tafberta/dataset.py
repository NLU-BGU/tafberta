from typing import List, Tuple, Generator, Union, Optional, Dict
import random
from itertools import combinations
import numpy as np
import pyprind
import torch
from collections import namedtuple

from transformers.models.roberta import RobertaTokenizer, RobertaTokenizerFast

from tokenizers import Encoding
from tokenizers import Tokenizer


from tafberta import configs
from tafberta.params import Params

from typing import List, Tuple, Generator, Union, Optional, Dict
import random
from itertools import combinations
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.models.roberta import RobertaTokenizer, RobertaTokenizerFast
from tokenizers import Encoding
from tokenizers import Tokenizer


def smart_tokenize(tokenizer: Union[Tokenizer, RobertaTokenizer],
                   sequence: str,
                   ) -> List[str]:

    # used by tafberta
    if isinstance(tokenizer, Tokenizer):
        tokens = tokenizer.encode(sequence, add_special_tokens=False).tokens

    # used pre-trained roberta-base
    elif isinstance(tokenizer, RobertaTokenizer) or isinstance(tokenizer, RobertaTokenizerFast):
        tokens = tokenizer.tokenize(sequence)  # does not add special tokens

    else:
        print(type(tokenizer))
        raise AttributeError('Unknown tokenizer')
    return tokens


def smart_encode(tokenizer: Union[Tokenizer, RobertaTokenizer],
                 sequences_in_batch: List[str],
                 ) -> List[Encoding]:

    # used by tafberta
    if isinstance(tokenizer, Tokenizer):
        # encodings = tokenizer.encode_batch(sequences_in_batch)  #change
        encodings = tokenizer.encode(sequences_in_batch)
        
    # used by pretrained roberta-base
    elif isinstance(tokenizer, RobertaTokenizer) or isinstance(tokenizer, RobertaTokenizerFast):
        tmp = tokenizer(sequences_in_batch, padding='longest', is_split_into_words=False)
        encodings = []
        Encoding_ = namedtuple('Encoding', ['ids', 'attention_mask'])
        for ids, am in zip(tmp['input_ids'], tmp['attention_mask']):
            encodings.append(Encoding_(ids, am))

    else:
        print(type(tokenizer))
        raise AttributeError('Unknown tokenizer')

    return encodings


class ProbingParams:
    sample_with_replacement = False
    max_input_length = 256
    leave_unmasked_prob_start = 0.0
    leave_unmasked_prob = 0.0
    random_token_prob = 0.0
    consecutive_masking = None
    mask_pattern_size = None
    num_mask_patterns = None
    allow_truncated_sentences = False
    batch_size = 32


class DataSet(Dataset):
    @classmethod
    def for_probing(cls,
                    sequences: List[str],
                    tokenizer: Union[Tokenizer, RobertaTokenizer],
                    ):
        def _get_mask_pattern_from_probing_sequence(sequence: str) -> Tuple[int]:
            tokens = smart_tokenize(tokenizer, sequence)
            res = [i for i, token in enumerate(tokens)
                   if token.endswith(configs.Data.mask_symbol)]
            return tuple(res)

        data = list(zip(sequences, [_get_mask_pattern_from_probing_sequence(s) for s in sequences]))
        return cls(sequences, tokenizer, ProbingParams(), data)

    def __init__(self,
                 sequences: List[str],
                 tokenizer: Union[Tokenizer, RobertaTokenizer],
                 params: Union[Params, ProbingParams],
                 data: Optional[List[Tuple[str, Tuple[int]]]] = None,
                 disallow_sub_words_when_probing: bool = False,
                 ):
        super().__init__()  # Call to Dataset's constructor
        self._sequences = sequences
        self.tokenizer = tokenizer
        self.params = params

        self.vocab_size = len(self.tokenizer.get_vocab())
        self.disallow_sub_words_when_probing = disallow_sub_words_when_probing

        assert 0.0 <= self.params.leave_unmasked_prob_start < 1.0
        assert self.params.leave_unmasked_prob_start <= self.params.leave_unmasked_prob <= 1.0
        assert 0.0 <= self.params.random_token_prob <= 1.0

        if not self._sequences:
            print(f'WARNING: No sequences passed to {self}.')
            self.data = None
            return

        # Compute tokenized sequence lengths and filter sequences
        self.tokenized_sequence_lengths, self._sequences = self._get_tokenized_sequence_lengths()

        # Prepare data if not provided
        self.data = data if data else list(self._gen_sequences_and_mask_patterns())
        if not self.params.consecutive_masking:
            random.shuffle(self.data)

        # Prepare number of batches
        if self.params.sample_with_replacement:
            self.num_batches = len(self.data) // self.params.batch_size
        else:
            self.num_batches = len(list(range(0, len(self.data), self.params.batch_size)))

        # Set up unmasking probabilities
        self.leave_unmasked_probabilities = iter(np.linspace(params.leave_unmasked_prob_start,
                                                             params.leave_unmasked_prob,
                                                             self.num_batches))

        # Initialize the weights for random token replacement
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initializes the weights for random token replacement.
        Weights should give higher probabilities to valid tokens and exclude special tokens.
        """
        weights = np.ones(self.vocab_size)
        num_special_tokens = 6  # Assuming the first few tokens in the vocab are special tokens
        weights[:num_special_tokens] = 0  # Set special tokens' weights to zero
        return weights / weights.sum()  # Normalize the weights to sum to 1

    def _get_tokenized_sequence_lengths(self) -> Tuple[List[int], List[str]]:
        """
        This function returns tokenized sequence lengths and filtered sequences based on max length.
        """
        tokenized_sequence_lengths = []
        filtered_sequences = []

        num_too_large = 0
        num_tokens_total = 0

        for s in self._sequences:
            tokens = smart_tokenize(self.tokenizer, s)
            num_tokens = len(tokens)

            # Check that words in probing sentences are never split into sub-words
            if self.disallow_sub_words_when_probing and isinstance(self.params, ProbingParams):
                if num_tokens != len(s.split()):
                    print(s)
                    print(tokens)
                    raise RuntimeError('Sub-tokens are not allowed in test sentences.')

            # Exclude sequence if it has too many tokens
            num_tokens_and_special_symbols = num_tokens + 2  # Including BOS and EOS tokens
            if not self.params.allow_truncated_sentences and num_tokens_and_special_symbols > self.params.max_input_length:
                num_too_large += 1
                continue

            num_tokens_total += num_tokens
            num_tokens_after_truncation = min(self.params.max_input_length - 2, num_tokens)
            tokenized_sequence_lengths.append(num_tokens_after_truncation)
            filtered_sequences.append(s)

        if self.params.allow_truncated_sentences:
            print(f'Did not exclude sentences because truncated sentences are allowed.')
        else:
            print(f'Excluded {num_too_large} sequences with more than {self.params.max_input_length} tokens.')

        print(f'Mean number of tokens in sequence={num_tokens_total / len(filtered_sequences):.2f}', flush=True)

        return tokenized_sequence_lengths, filtered_sequences

    def __len__(self) -> int:
        """
        Returns the total number of data points.
        """
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Union[torch.LongTensor, None], torch.Tensor]:
        """
        Retrieve a single sample based on the index.
        """
        if self.data is None or len(self.data) == 0:
            raise IndexError("Trying to access an item from an empty dataset.")
    
        # Get the sequence and mask pattern at the specified index
        try:
            sequence, mask_pattern = self.data[idx]
        except IndexError as e:
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.data)}. Error: {e}")
    
        # Tokenize the single sequence
        encoding = smart_encode(self.tokenizer, sequence)  # Tokenize the single sequence, returning a list
    
        # Mask the single input
        x, y, mask = self.mask_input_ids([encoding], [mask_pattern])  # Pass as lists
    
        # Return the input_ids, attention_mask, and labels for the single sample
        return x, y, mask

    def _gen_sequences_and_mask_patterns(self) -> Generator[Tuple[str, Tuple[int]], None, None]:
        """
        Generates sequences and their corresponding mask patterns.
        """
        pbar = pyprind.ProgBar(len(self._sequences))  # Corrected: Use _sequences
        for s, num_tokens_after_truncation in zip(self._sequences, self.tokenized_sequence_lengths):
            for mp in self._gen_make_mask_patterns(num_tokens_after_truncation):
                yield s, mp
            pbar.update()

    def mask_input_ids(self,
                       batch_encoding: List[Encoding],
                       mask_patterns: List[Tuple[int]],
                       ) -> Tuple[Dict[str, torch.tensor], Union[torch.LongTensor, None], torch.tensor]:

        # collect each encoding into a single matrix
        input_ids_raw = np.array([e.ids for e in batch_encoding])
        attention_mask = np.array([e.attention_mask for e in batch_encoding])

        batch_shape = input_ids_raw.shape
        mask = self._make_mask_matrix(batch_shape, mask_patterns)

        if batch_shape[1] > self.params.max_input_length:
            raise ValueError(f'Batch dim 1 ({batch_shape[1]}) is larger than {self.params.max_input_length}')

        # decide unmasking and random replacement
        leave_unmasked_prob = next(self.leave_unmasked_probabilities)
        rand_or_unmask_prob = self.params.random_token_prob + leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(*batch_shape) < rand_or_unmask_prob)
            if self.params.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(*batch_shape) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        # mask_insertion_matrix
        if unmask is not None:
            mask_insertion_matrix = mask ^ unmask  # XOR: True in mask will make True in mask_insertion False
        else:
            mask_insertion_matrix = mask

        # insert mask symbols - this has no effect during probing
        if np.any(mask_insertion_matrix):
            input_ids = np.where(mask_insertion_matrix,
                                 self.tokenizer.token_to_id(configs.Data.mask_symbol),
                                 input_ids_raw)
        else:
            input_ids = np.copy(input_ids_raw)

        # insert random tokens
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                input_ids[rand_mask] = np.random.choice(self.vocab_size, num_rand, p=self.weights)

        x = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
        }

        if not mask_patterns:  # forced-choice probing
            y = None
        else:
            y = torch.tensor(input_ids_raw[mask]).requires_grad_(False)

        return x, y, torch.tensor(mask)

    def _gen_make_mask_patterns(self,
                                num_tokens_after_truncation: int,
                                ) -> Generator[Tuple[int], None, None]:
        """
        make a number of mask patterns that is as large as possible to the requested number.

        a mask_pattern is a tuple of 1 or more integers
        corresponding to the indices of a tokenized sequence that should be masked.

        notes:
        - pattern size is dynamically shortened if a tokenized sequence is smaller than mask_pattern_size.
        - num_mask_patterns is dynamically adjusted if number of possible patterns is smaller than num_mask_patterns.
        """
        random.seed(None)  # use different patterns across different runs

        pattern_size = min(self.params.mask_pattern_size, num_tokens_after_truncation)

        # sample patterns from population of all possible patterns
        all_mask_patterns = list(combinations(range(num_tokens_after_truncation), pattern_size))
        num_patterns = min(self.params.num_mask_patterns, len(all_mask_patterns))

        # generate mask patterns that are unique
        predetermined_patterns = iter(random.sample(all_mask_patterns, k=num_patterns))

        num_yielded = 0
        while num_yielded < num_patterns:

            if self.params.probabilistic_masking:
                if self.params.mask_probability == 'auto':
                    prob = self.params.mask_pattern_size / num_tokens_after_truncation
                elif isinstance(self.params.mask_probability, float) and 0 < self.params.mask_probability < 1:
                    prob = self.params.mask_probability
                else:
                    raise AttributeError('invalid arg to mask_probability')
                mask_pattern = tuple([i for i in range(num_tokens_after_truncation) if random.random() < prob])
            else:
                mask_pattern = next(predetermined_patterns)

            if mask_pattern:
                num_yielded += 1
            else:
                continue  # pattern can be empty when sampling probabilistically

            yield mask_pattern

    @staticmethod
    def _make_mask_matrix(batch_shape: Tuple[int, int],
                          mask_patterns: List[Tuple[int]],
                          ) -> np.array:
        """
        return matrix specifying which tokens in a batch should be masked (but not necessarily replaced by mask symbol).

        notes:
        - mask_patterns is based on tokens without special symbols (eos, bos), so conversion must be done
        """
        res = np.zeros(batch_shape, dtype=np.bool_)
        assert batch_shape[0] == len(mask_patterns)
        for row_id, mask_pattern in enumerate(mask_patterns):
            # a mask pattern may consist of zero, one, or more than one index (of a token to be masked)
            for mi in mask_pattern:
                col_id = mi + 1  # handle BOS symbol
                res[row_id, col_id] = True
        return res
