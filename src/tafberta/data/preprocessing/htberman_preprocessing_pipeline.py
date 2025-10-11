"""
HTBerman end-to-end prepare script
=================================

Runs the two preprocessing phases in order:
  1) Utterance-level matching (align files by sorted index; write JSONL with %mor)
  2) Token-level matching (heuristic token edits; DictaBERT segmentation if available)

Default I/O (from `tafberta.configs`):
  processed/htberman/aligned_pairs.jsonl
  processed/htberman/corpus_utt_aligned.txt
  processed/htberman/aligned_pairs_token.jsonl
  processed/htberman/corpus_utt_token_aligned.txt
  processed/htberman/details_token.parquet

CLI examples
------------
# full pipeline, default locations
python -m tafberta.data.preprocessing.htberman_prepare

# keep only selected speakers in utterance stage (e.g., caregivers)
python -m tafberta.data.preprocessing.htberman_prepare \
  --keep-speakers MOT,FAT,ADU,GM,GF

# exclude speakers in token stage (CDS only by default excludes CHI)
python -m tafberta.data.preprocessing.htberman_prepare \
  --remove-speakers CHI

# run only one phase
python -m tafberta.data.preprocessing.htberman_prepare --only-utterance
python -m tafberta.data.preprocessing.htberman_prepare --only-token
"""
from pathlib import Path
from typing import Optional, Tuple
import argparse
import sys

from tafberta import configs

from tafberta.data.preprocessing.htberman_utterance_matcher import main as utterance_main
from tafberta.data.preprocessing.htberman_token_matcher import main as token_main


# -----------------------
# helpers
# -----------------------

def _to_tuple(opt: Optional[str]) -> Optional[Tuple[str, ...]]:
    if opt is None:
        return None
    opt = opt.strip()
    if not opt:
        return None
    # support comma/space separated
    parts = [p.strip() for p in opt.replace(" ", ",").split(",") if p.strip()]
    return tuple(parts) if parts else None


# -----------------------
# main driver
# -----------------------

def run_pipeline(
    *,
    outdir: Optional[Path] = None,
    keep_speakers: Optional[str] = None,
    remove_speakers: Optional[str] = "CHI",
    only_utterance: bool = False,
    only_token: bool = False,
) -> None:
    outdir = outdir or (configs.Dirs.processed / "htberman")
    outdir.mkdir(parents=True, exist_ok=True)

    # Phase 1 outputs
    pairs_jsonl = outdir / "aligned_pairs.jsonl"
    heb_txt = outdir / "corpus_utt_aligned.txt"

    # Phase 2 outputs
    pairs_token_jsonl = outdir / "aligned_pairs_token.jsonl"
    heb_token_txt = outdir / "corpus_utt_token_aligned.txt"
    details_parquet = outdir / "details_token.parquet"

    keep_speakers = _to_tuple(keep_speakers)
    remove_speakers = _to_tuple(remove_speakers) or tuple()

    print("[prepare] OUTDIR:", outdir)
    print("[prepare] keep_speakers:", keep_speakers)
    print("[prepare] remove_speakers:", remove_speakers)

    # ----------------
    # Phase 1
    # ----------------
    if not only_token:
        print("[prepare] Phase 1: utterance-level matching…")
        # Support both signatures: (main) or (build_htberman)
        try:
            # htb_utterance_matcher.main signature
            utterance_main(
                out_pairs_jsonl=pairs_jsonl,
                out_hebrew_txt=heb_txt,
                keep_speakers=keep_speakers,
            )
        except TypeError:
            # builder signature returns the paths
            _pairs, _txt = utterance_main(
                out_pairs_jsonl=pairs_jsonl,
                out_hebrew_txt=heb_txt,
                keep_speakers=keep_speakers,
            )
        print("[prepare] Phase 1 done →", pairs_jsonl)

    # ----------------
    # Phase 2
    # ----------------
    if not only_utterance:
        print("[prepare] Phase 2: token-level matching…")
        token_main(
            in_pairs_jsonl=pairs_jsonl,
            out_pairs_token_jsonl=pairs_token_jsonl,
            out_hebrew_token_txt=heb_token_txt,
            details_parquet=details_parquet,
            remove_speakers=remove_speakers,
        )
        print("[prepare] Phase 2 done →", pairs_token_jsonl)

    print("[prepare] ALL DONE ✅")


# -----------------------
# CLI
# -----------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTBerman utterance+token pipeline")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output directory (default: configs.Dirs.processed/htberman)")
    p.add_argument("--keep-speakers", type=str, default=None,
                   help="Comma/space-separated list of speakers to KEEP in Phase 1 (e.g., 'MOT,FAT,ADU')")
    p.add_argument("--remove-speakers", type=str, default="CHI",
                   help="Comma/space-separated list of speakers to REMOVE in Phase 2 (default: 'CHI' for CDS only)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-utterance", action="store_true", help="Run only Phase 1")
    g.add_argument("--only-token", action="store_true", help="Run only Phase 2 (expects Phase 1 outputs exist)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(
        outdir=args.outdir,
        keep_speakers=args.keep_speakers,
        remove_speakers=args.remove_speakers,
        only_utterance=args.only_utterance,
        only_token=args.only_token,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
