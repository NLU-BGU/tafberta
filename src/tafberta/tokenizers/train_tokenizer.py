from typing import List, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevel_decoder
from tokenizers.pre_tokenizers import ByteLevel as ByteLevel_pre_tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import RobertaProcessing

from tafberta import configs


def train_bpe_tokenizer(
    corpus_files: Optional[List[str]] = None,
    vocab_size: Optional[int] = None,
    min_frequency: Optional[int] = None,
    unk_token: Optional[str] = None,
    special_tokens: Optional[List[str]] = None,
    add_prefix_space: Optional[bool] = None,
    max_length: int = 128,
    save_path: Optional[str] = None,
) -> Tokenizer:
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
        Defaults to configs.Tokenizer.add_prefix_space.
    max_length : int, optional
        Max sequence length for truncation (default: 128).
    save_path : str, optional
        Where to save the trained tokenizer. Defaults to configs.Dirs.tokenizers.tokenizer_path.


    Returns
    -------
    Tokenizer
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
        else configs.Tokenizer.add_prefix_space
    )
    save_path = save_path or str(configs.Tokenizer.tokenizer_path)

    # core model + pre-tokenizer
    model = BPE(unk_token=unk_token)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevel_pre_tokenizer(add_prefix_space=add_prefix_space)

    # trainer
    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    tokenizer.train(files=corpus_files, trainer=trainer)

    # add additional info
    tokenizer.post_processor = RobertaProcessing(sep=('</s>', tokenizer.token_to_id("</s>")),
                                                cls=('<s>', tokenizer.token_to_id("<s>")),
                                                add_prefix_space=add_prefix_space)
    tokenizer.decoder = ByteLevel_decoder(add_prefix_space=add_prefix_space)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(configs.Data.pad_symbol),
        pad_token=configs.Data.pad_symbol
        )
    tokenizer.enable_truncation(max_length=max_length)

    # save tokenizer
    tokenizer.save(save_path, pretty=True)

    print(f'Saved tokenizer config to {save_path}')
