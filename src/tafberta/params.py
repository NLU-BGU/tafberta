from dataclasses import dataclass
from typing import Tuple


# the default model is the best model trained on just HTBerman, with unmasking
param2default = {
    # data
    'sample_with_replacement': False,  # this must be False if corpus order is to be preserved during training
    'training_order': 'original',  # original or shuffled, use this alongside consecutive_masking=True
    'consecutive_masking': False,  # better dev pp and grammatical accuracy when false
    'num_sentences_per_input': 1,  # if too large -> may exceed CUDA memory, 1 is best for good number-agreement
    'include_punctuation': True,
    'allow_truncated_sentences': False,
    'num_mask_patterns': 10,  # 10 is better than fewer
    'mask_pattern_size': 2,  # used only if probabilistic_masking = False
    'probabilistic_masking': True,
    'mask_probability': 0.15,  # used only if probabilistic_masking = true
    'leave_unmasked_prob_start': 0.1,
    'leave_unmasked_prob': 0.1,
    'random_token_prob': 0.1,
    'corpora': ('htberman',),
    'tokenizer': 'bermanLong_CDS',
    # 'corpora': ('wikipedia_subset',),
    # 'tokenizer': 'wikipedia',
    'add_prefix_space': True,  # better if True, whether to treat first token like any other token (False in GPT-2)
    'max_input_length': 128,  # unacceptable performance if lower than ~32

    # training
    'batch_size': 128,
    'lr': 1e-4,  # 1e-4 is used in fairseq (and performs better here), and 1e-3 is default in huggingface
    'num_epochs': 100,  # use 1 epoch to use dynamic masking
    'num_warmup_steps': 24_000,  # 24K used in Roberta-base
    'weight_decay': 0.0,
    'seed': 1,

    # model
    'hidden_size': 64,
    'num_layers': 10,
    'num_attention_heads': 4,
    'intermediate_size': 2048,
    'initializer_range': 0.02,  # stdev of trunc normal for initializing all weights
    'layer_norm_eps': 1e-5,  # 1e-5 default in fairseq (and slightly better performance), 1e-12 default in hgugingface,

    #save path
    'save_to_folder_by_epoch': True,
}


@dataclass
class Params:
    """
    this object is loaded at the start of job.main() by calling Params.from_param2val(),
    and is populated by Ludwig with hyper-parameters corresponding to a single job.
    """

    # data
    sample_with_replacement: bool
    consecutive_masking: bool
    training_order: str
    num_sentences_per_input: int
    include_punctuation: bool
    allow_truncated_sentences: bool
    num_mask_patterns: int
    mask_pattern_size: int
    probabilistic_masking: bool
    mask_probability: float
    leave_unmasked_prob_start: float
    leave_unmasked_prob: float
    random_token_prob: float
    corpora: Tuple[str]
    tokenizer: str
    add_prefix_space: bool
    max_input_length: int

    # training
    batch_size: int
    lr: float
    num_epochs: int
    num_warmup_steps: int
    weight_decay: float
    seed: int

    # model
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    initializer_range: float
    layer_norm_eps: float
    save_to_folder_by_epoch: bool

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'project_path', 'save_path']}
        return cls(**kwargs)