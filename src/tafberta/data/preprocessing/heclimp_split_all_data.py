from pathlib import Path
from typing import List, Tuple, Sequence, Optional
import argparse
import random
import sys

from tafberta import configs


PARADIGM = "agreement_determiner_noun-across_0_adjective"
DEFAULT_DIRS = (
    (getattr(configs.Dirs, "heclimp_testsuits_htberman", configs.Dirs.processed / "heclimp"))
    / PARADIGM
).parent  # base folder that contains both *_gen and *_num


def _default_target_dirs() -> List[Path]:
    base = getattr(configs.Dirs, "heclimp_testsuits_htberman",
                   configs.Dirs.processed / "heclimp" / "testsuites")
    return [
        base / f"{PARADIGM}_gen",
        base / f"{PARADIGM}_num",
    ]


def _read_lines(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _write_lines(p: Path, lines: Sequence[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def _pairs_from_lines(lines: List[str]) -> List[Tuple[str, str]]:
    if len(lines) % 2 != 0:
        # Drop last dangling line to preserve pairing if needed
        lines = lines[:-1]
    return list(zip(lines[0::2], lines[1::2]))


def split_all_data_dir(
    dir_path: Path,
    *,
    dev_ratio: float = 0.20,
    seed: int = 42,
    shuffle: bool = False,
) -> Tuple[Path, Path]:
    """
    Split {dir_path}/all_data.txt into pair-preserving dev/test splits.
    - dev: ~dev_ratio of pairs
    - test: remainder (≈80% when dev_ratio=0.2)

    Returns: (dev_path, test_path)
    """
    all_path = dir_path / "all_data.txt"
    if not all_path.exists():
        raise FileNotFoundError(f"all_data.txt not found at: {all_path}")

    lines = _read_lines(all_path)
    pairs = _pairs_from_lines(lines)
    n = len(pairs)
    if n == 0:
        raise ValueError(f"No pairs found in {all_path} (need even number of lines).")

    k = int(round(n * dev_ratio))
    # keep both splits non-empty where possible
    if k <= 0 and n > 1:
        k = 1
    if k >= n:
        k = n - 1

    idx = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)

    dev_idx = set(idx[:k])
    dev_pairs = [pairs[i] for i in range(n) if i in dev_idx]
    test_pairs = [pairs[i] for i in range(n) if i not in dev_idx]

    dev_lines = [x for pair in dev_pairs for x in pair]
    test_lines = [x for pair in test_pairs for x in pair]

    dev_path = dir_path / "dev.txt"
    test_path = dir_path / "test.txt"
    _write_lines(dev_path, dev_lines)
    _write_lines(test_path, test_lines)

    print(f"[split] {dir_path}")
    print(f"        total pairs: {n}")
    print(f"        -> dev.txt  : {len(dev_pairs)} pairs ({len(dev_lines)} lines)")
    print(f"        -> test.txt : {len(test_pairs)} pairs ({len(test_lines)} lines)")
    return dev_path, test_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split HeCLiMP all_data.txt into dev/test (pair-preserving).")
    p.add_argument(
        "--dirs",
        type=Path,
        nargs="*",
        default=None,
        help="One or more directories that contain all_data.txt. "
             "If omitted, defaults to the two standard paradigm dirs (…_gen and …_num).",
    )
    p.add_argument("--dev-ratio", type=float, default=0.20, help="Proportion of pairs for dev (default: 0.20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--shuffle", type=bool, default=False, help="Shuffle pairs before splitting")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    target_dirs = args.dirs or _default_target_dirs()

    for d in target_dirs:
        split_all_data_dir(d, dev_ratio=args.dev_ratio, seed=args.seed, shuffle=args.shuffle)


if __name__ == "__main__":
    main(sys.argv[1:])
