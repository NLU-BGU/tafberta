# src/tafberta/data/preprocessing/htberman_builder.py
from pathlib import Path
from typing import Dict, Iterator, Iterable, List, Tuple, Optional
import os
import json
import pickle

import pylangacq  # for CHAT reading
from tafberta import configs

"""
UTTERANCE-LEVEL MATCHING (Phase 1)
----------------------------------
This module performs ONLY the utterance-level alignment between the
latin-based CHILDES (BermanLong) transcripts and the standard-Hebrew
files prepared in Phase 0 (file-level prep).

Important alignment policy:
- We **sort** the list of English CHAT files by file name and the list
  of Hebrew text files by file name, then match them **by index**.
  We do NOT rely on file names being identical.
- For files with different line counts, we drop specific indices from
  the longer side (based on a manually curated list below) so they
  become length-aligned for a clean zip().

Outputs:
- aligned_pairs.jsonl  (per-utterance JSON records; both sides + speaker)
- corpus_utt_aligned.txt (only the Hebrew sentence per aligned pair)

Token-level normalization is NOT done here. That will be Phase 2.
"""

# -------------------------
# Readers / sources loading
# -------------------------

def load_chat_readers() -> Dict[str, "pylangacq.Reader"]:
    """Load CHILDES readers mapping for BermanLong.

    Preferred behavior:
    - If a cache exists at `Dirs.htberman_raw_english/readers_BermanLong.p`,
      load it for reproducibility.
    - Otherwise, build it from CHILDES using `pylangacq.read_chat`, drop the
      known empty/problematic file, and save the cache for next runs.
    """
    cache = configs.Dirs.htberman_raw_english / "readers_BermanLong.p"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    # Build fresh mapping
    reader = pylangacq.read_chat(configs.DataPrep.url % configs.DataPrep.childes_in_htberman_dataset)
    readers = dict(zip(reader.file_paths(), [reader.pop_left() for _ in range(len(reader.file_paths()))]))
    # remove known empty/problematic file if present
    readers.pop("BermanLong/Leor/leo300e.cha", None)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(readers, f, protocol=pickle.HIGHEST_PROTOCOL)

    return readers


def sorted_english_files(readers: Dict[str, "pylangacq.Reader"]) -> List[Tuple[str, "pylangacq.Reader"]]:
    """Sort CHAT files by their base filename; return list of (file_key, reader)."""
    return tuple(readers.items())
    return sorted(readers.items(), key=lambda kv: os.path.basename(kv[0]))


def sorted_hebrew_files(hebrew_dir: Path) -> List[Path]:
    """Return sorted list of Hebrew speech file paths (post file-level prep)."""
    hebrew_dir = Path(hebrew_dir)
    files = [hebrew_dir / fn for fn in os.listdir(hebrew_dir) if fn.endswith(".txt")]
    files.sort(key=lambda p: p.name)
    return files


# --------------------------------------
# Manual per-file extra-lines annotations
# --------------------------------------
# These lists correspond to *the order of files among those with diffs*.
# We first compute details for every ENG/HEB file pair, then build
# diff_files = [det for det in details if det['diff_heb_eng'] != 0]
# The keys below (0..63) index into that diff_files list.
MANUAL_EXTRA_LINES_BY_DIFF_INDEX: Dict[int, List[int]] = {
    0: [137],
    1: [116, 123],
    2: [41],
    3: [381],
    4: [109],
    5: [192, 317, 318, 319],
    6: [140, 464],
    7: [662],
    8: [333],
    9: [247, 249, 252],
    10: [70],
    11: [57],
    12: [25],
    13: [309],
    14: [335],
    15: [871, 917, 919, 932, 934, 983, 985, 987, 989, 991, 1162, 1164, 1166, 1196],
    16: [368],
    17: [350, 372, 374, 379, 387, 544, 546, 548, 550, 621],
    18: [129, 252, 265],
    19: [147],
    20: [127],
    21: [43],
    22: [211, 240],
    23: [387],
    24: [245, 246],
    25: [169],
    26: [307],
    27: [5],
    28: [100, 101],
    29: [804, 811, 849],
    30: [174],
    31: [0],
    32: [314],
    33: [526, 530],
    34: [33],
    35: [355],
    36: [781],
    37: [232, 233],
    38: [352],
    39: [154],
    40: [311, 312, 316, 320],
    41: [229],
    42: [169],
    43: [562],
    44: [140],
    45: [461],
    46: [405],
    47: [259],
    48: [60, 95, 266, 406, 493],
    49: [37],
    50: [442],
    51: [70, 71],
    52: [188],
    53: [80, 145],
    54: [371, 474],
    55: [102],
    56: [384],
    57: [86, 87, 117, 118, 123, 442],
    58: [8, 24, 164, 415],
    59: [241],
    60: [88, 212],
    61: [48, 49, 464],
    62: [12, 204, 277],
    63: [17, 433, 483],
}


# -----------------------
# Utility: drop by indices
# -----------------------

def del_list_by_ids(seq: List, ids: Iterable[int]) -> List:
    ids_set = set(ids)
    if not ids_set:
        return list(seq)
    return [item for j, item in enumerate(seq) if j not in ids_set]


# ---------------------------
# Details computation & diffs
# ---------------------------

def compute_file_details(
    readers: Dict[str, "pylangacq.Reader"],
    hebrew_dir: Path,
) -> List[Dict[str, object]]:
    """
    Pair English CHAT files with Hebrew files by **sorted index**, then compute
    counts and diffs. Returns a list of dicts with metadata per pair.
    """
    eng_sorted = sorted_english_files(readers)
    heb_sorted = sorted_hebrew_files(hebrew_dir)

    if len(heb_sorted) != len(eng_sorted):
        print(
            f"[warn] #Heb files ({len(heb_sorted)}) != #Eng files ({len(eng_sorted)}). Proceeding by min length."
        )

    n = min(len(eng_sorted), len(heb_sorted))
    details: List[Dict[str, object]] = []

    for i in range(n):
        eng_key, eng_reader = eng_sorted[i]
        heb_path = heb_sorted[i]

        # Count utterances in ENG file
        try:
            eng_utts = eng_reader.utterances()
            eng_count = len(eng_utts)
        except Exception:
            # fallback
            transcripts = getattr(eng_reader, "transcripts", lambda: [])()
            eng_count = len(transcripts)

        # Count lines in Hebrew file
        with open(heb_path, "r", encoding="utf-8") as f:
            heb_lines = [ln for ln in f.readlines()]  # keep raw; drop/strip later
        heb_count = len(heb_lines)

        details.append(
            {
                "pair_index": i,
                "eng_file_path": eng_key,
                "heb_file_path": str(heb_path),
                "eng_utt_count": eng_count,
                "heb_line_count": heb_count,
                "diff_heb_eng": heb_count - eng_count,
                "extra_lines_ids": [],  # to be filled for diff files
            }
        )

    # Fill manual extra-lines for those with diffs, based on their order among diffs
    diff_files = [d for d in details if d["diff_heb_eng"] != 0]
    for idx_among_diffs, det in enumerate(diff_files):
        ids = MANUAL_EXTRA_LINES_BY_DIFF_INDEX.get(idx_among_diffs)
        if ids:
            det["extra_lines_ids"] = ids

    return details


# ------------------------------------
# Alignment iterator over paired files
# ------------------------------------

def iter_aligned_pairs(
    details: List[Dict[str, object]],
    readers: Dict[str, "pylangacq.Reader"],
) -> Iterator[Dict[str, object]]:
    """
    Yield aligned utterance pairs after dropping manual extra lines on the
    longer side so that lengths match. Each yield is a JSON-serializable dict
    with metadata + english transcript + hebrew sentence.
    """
    for det in details:
        eng_key = det["eng_file_path"]  # type: ignore
        heb_path = Path(det["heb_file_path"])  # type: ignore
        diff = int(det["diff_heb_eng"])  # heb - eng
        extra_ids: List[int] = list(det.get("extra_lines_ids", []))  # type: ignore

        reader = readers[eng_key]  # type: ignore
        try:
            eng_utts = reader.utterances()
        except Exception:
            # minimal fallback: build pseudo-utterances from transcripts
            transcripts = getattr(reader, "transcripts", lambda: [])()
            eng_utts = [{"tokens": [{"word": w} for w in t.split()], "participant": ""} for t in transcripts]

        with open(heb_path, "r", encoding="utf-8") as f:
            heb_lines = [ln.rstrip("\n") for ln in f]

        # Drop from the longer side
        if diff <= 0:
            # Hebrew has <= lines than English: drop extra English utterances
            eng_utts_clean = del_list_by_ids(eng_utts, extra_ids)
            heb_lines_clean = heb_lines
        else:
            # Hebrew has more lines: drop extra Hebrew lines
            eng_utts_clean = eng_utts
            heb_lines_clean = del_list_by_ids(heb_lines, extra_ids)

        m = min(len(eng_utts_clean), len(heb_lines_clean))
        for i in range(m):
            utt = eng_utts_clean[i]
            heb = heb_lines_clean[i].strip()
            if not heb:
                continue

            # extract speaker + english surface
            speaker = (utt.get("participant") if isinstance(utt, dict) else getattr(utt, "participant", "")) or ""
            # tokens = (utt.get("tokens") if isinstance(utt, dict) else getattr(utt, "tokens", [])) or []
            # if tokens and isinstance(tokens[0], dict) and "word" in tokens[0]:
            #     eng_surface = " ".join(tok.get("word", "") for tok in tokens)
            # else:
            #     # best-effort fallback
            #     eng_surface = getattr(utt, "transcript", "") or ""

            sentence_eng = ' '.join([token.word for token in utt.tokens])
            yield {
                "eng_file_path": eng_key,
                "heb_file_path": str(heb_path),
                "utt_index": i,
                "speaker": speaker,
                "english": sentence_eng,
                "hebrew": heb,
            }


# -------------------------
# Builder entrypoint (Phase 1)
# -------------------------

def build_htberman(
    *,
    out_pairs_jsonl: Optional[Path] = None,
    out_hebrew_txt: Optional[Path] = None,
    keep_speakers: Optional[Tuple[str, ...]] = None,
    remove_speakers: Optional[Tuple[str, ...]] = None
) -> Tuple[Path, Path]:
    """
    Utterance-level alignment only.

    - Pairs ENG/HEB files by **sorted index** (not by names),
    - Applies manual extra-line removals per diff file,
    - Emits aligned records + a plain-text Hebrew corpus.

    Args:
        out_pairs_jsonl: where to write JSONL pairs (defaults under processed/htberman)
        out_hebrew_txt: where to write Hebrew-only lines (defaults under processed/htberman)
        keep_speakers: optional tuple of speakers to KEEP (e.g., ("MOT","FAT","ADU","GM","GF")).
                       If None, keep all.
        remove_speakers: optional tuple of speakers to REMOVE (e.g., ("CHI")).
                         If None, no speakers are removed.
    Returns:
        (pairs_jsonl_path, hebrew_txt_path)
    """
    readers = load_chat_readers()
    details = compute_file_details(readers, configs.Dirs.htberman_processed_hebrew)

    if out_pairs_jsonl is None:
        out_pairs_jsonl = configs.Dirs.processed / "htberman" / "aligned_pairs.jsonl"
    if out_hebrew_txt is None:
        out_hebrew_txt = configs.Dirs.processed / "htberman" / "corpus_utt_aligned.txt"

    out_pairs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_hebrew_txt.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_lines = 0

    with open(out_pairs_jsonl, "w", encoding="utf-8") as jp, \
         open(out_hebrew_txt, "w", encoding="utf-8") as ht:
        for rec in iter_aligned_pairs(details, readers):
            if remove_speakers and rec.get("speaker") in remove_speakers:
                continue
            if keep_speakers and rec.get("speaker") not in keep_speakers:
                continue
            jp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ht.write(rec["hebrew"] + "\n")
            n_pairs += 1
            n_lines += 1

    print(f"[utterance-level] wrote {n_pairs} JSONL records -> {out_pairs_jsonl}")
    print(f"[utterance-level] wrote {n_lines} Hebrew lines -> {out_hebrew_txt}")

    return out_pairs_jsonl, out_hebrew_txt


if __name__ == "__main__":
    # build_htberman()
    build_htberman(remove_speakers=("CHI",))
