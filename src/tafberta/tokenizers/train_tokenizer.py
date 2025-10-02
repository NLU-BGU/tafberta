from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevel_pre_tokenizer
from tokenizers.trainers import BpeTrainer

from tafberta import configs


def train_bpe_tokenizer(
    corpus_files=None,
    vocab_size=None,
    min_frequency=None,
    unk_token=None,
    special_tokens=None,
    add_prefix_space=None,
):
    """
    Train a BPE tokenizer with configurable parameters. By default, it uses
    the values defined in tafberta.configs.

    Parameters
    ----------
    corpus_files : list[str], optional
        List of paths to training text files. Defaults to configs.Tokenizer.corpus_file_paths.
    vocab_size : int, optional
        Vocabulary size. Defaults to configs.Tokenizer.vocab_size.
    min_frequency : int, optional
        Minimum frequency for tokens. Defaults to configs.Tokenizer.min_frequency.
    unk_token : str, optional
        Unknown token symbol. Defaults to configs.Data.unk_symbol.
    special_tokens : list[str], optional
        Special tokens list. Defaults to configs.Data.roberta_symbols.
    add_prefix_space : bool, optional
        Whether to add a prefix space in the ByteLevel pre-tokenizer.
        Defaults to configs.Tokenizer.ADD_PREFIX_SPACE.

    Returns
    -------
    tokenizer : Tokenizer
        A trained Hugging Face Tokenizer object.
    """

    # fallback to configs if not provided
    corpus_files = corpus_files or configs.Tokenizer.corpus_file_paths
    vocab_size = vocab_size or configs.Tokenizer.vocab_size
    min_frequency = min_frequency or configs.Tokenizer.min_frequency
    unk_token = unk_token or configs.Data.unk_symbol
    special_tokens = special_tokens or configs.Data.roberta_symbols
    add_prefix_space = (
        add_prefix_space if add_prefix_space is not None
        else configs.Tokenizer.ADD_PREFIX_SPACE
    )

    model = BPE(unk_token=unk_token)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevel_pre_tokenizer(
        add_prefix_space=add_prefix_space
        )

    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    tokenizer.train(files=corpus_files, trainer=trainer)

    return tokenizer
