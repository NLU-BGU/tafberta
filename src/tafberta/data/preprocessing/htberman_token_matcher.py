"""
TOKEN-LEVEL MATCHING (Phase 2)
==============================

• Consumes Phase‑1 output: `processed/htberman/aligned_pairs.jsonl`
• Produces token-aligned Hebrew (editing both Hebrew and English sides)
• Keeps CHILDES `%mor` (for HeCLiMP building)
• Filters to CDS only by default (excludes child utterances: speaker == "CHI")
• DictaBERT segmentation when available, safe fallback otherwise
• Pipeline order:
    0) **preprocess** Hebrew line (spacing around tags/punct; per user regex)
    1) segment with DictaBERT (or fallback spacing)
    2) merge children's names (Hagar/Leor/Lior)
    2.5) drop angle brackets `<` `>` (not present in ENG)
    3) manual joins & normalizations
    4) punctuation / prepositions / MWEs
    5) re-run prepositions (indexes may change)
    6) **insert CLITIC** at the very end (by ENG index)

Outputs under `configs.Dirs.processed/htberman/` by default:
    - aligned_pairs_token.jsonl
    - corpus_utt_token_aligned.txt
    - details_token.parquet
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

import json
import re
import pandas as pd

from tafberta import configs

# ---------------------------------------------------------------------------
# Line-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_heb_line(line: str) -> str:
    """
    Preprocess a single line of Hebrew text according to specific tokenization rules.

    This function splits the input line into tokens based on whitespace and applies the following rules:
        - If a token matches the pattern `<HEB_TAG>rest`, it is split into two tokens: `'<HEB_TAG>'` and `'rest'`.
        - If a token consists of a Hebrew word followed by non-digit, non-Hebrew,
        non-space characters (e.g., punctuation), it is split into the word and the trailing characters.
        - Otherwise, the token is left unchanged.

    Parameters
    ----------
    line : str
        The input line of text to preprocess.

    Returns
    -------
    str
        The preprocessed line, with tokens separated by spaces according to the specified rules.
    """
    line = line.strip()
    parts = line.split()
    out: List[str] = []

    for tok in parts:
        # Case 1: <HEB_TAG> + trailing chunk (e.g., "<שלום>!!!")
        m = re.match(r'(<[א-ת]*>)(.+)', tok)
        if m:
            tag, rest = m.groups()
            if tag:
                out.append(tag)
            if rest:
                out.append(rest)
            continue

        # Case 2: Hebrew word followed by non-(digit|Hebrew|space) chars
        m = re.match(r'([א-ת]+)([^\dא-ת\s]+)', tok)
        if m:
            word, punct = m.groups()
            if word:
                out.append(word)
            if punct:
                out.append(punct)
        else:
            out.append(tok)

    return " ".join(out)


def preprocess_eng_line(line: str) -> str:
    """
    Preprocesses an English line by removing the word 'POSTCLITIC' if present.

    Parameters
    ----------
    line : str
        The input string to preprocess.

    Returns
    -------
    str
        The preprocessed string with 'POSTCLITIC' removed if it was present.
    """
    if "POSTCLITIC" in line.split():
        line = line.replace("POSTCLITIC", "")
    return line


def preprocess_heb_file(file_path_in: str, file_path_out: str) -> None:
    """Batch version of preprocess_heb_line."""
    with open(file_path_in, "r", encoding="utf-8") as f_in, open(file_path_out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(preprocess_heb_line(line) + "\n")


# ---------------------------------------------------------------------------
# DictaBERT segmentation (with safe fallback)
# ---------------------------------------------------------------------------
_SEGMENT_WITH_DICTABERT = True
try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    _tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-seg')
    _model = AutoModel.from_pretrained('dicta-il/dictabert-seg', trust_remote_code=True)
    _model.eval()
    print("[segmenter] Loaded dicta-il/dictabert-seg (DictaBERT)")

    def _flatten(l):
        return [item for sublist in l for item in sublist]

    def segment_heb(sentence: str) -> str:
        out = _model.predict([sentence], _tokenizer)
        return ' '.join(_flatten(out[0][1:-1]))
except Exception:  # pragma: no cover
    _SEGMENT_WITH_DICTABERT = False
    print("[segmenter] Could not load DictaBERT; using fallback segmenter (spaced_punct)")

    def segment_heb(sentence: str) -> str:
        # very conservative fallback: just space punctuation
        return spaced_punct(sentence)


# ---------------------------------------------------------------------------
# Name merging per spec + resources
# ---------------------------------------------------------------------------
_HAGAR_RE = re.compile(r"\b(ה גרי|ה גרוש|ה גר|ה גרילי|ה גרולה|ה גרילה|ה גררי|ה גרילך|ה גרה)\b")

def correct_hagar(heb_sen: str) -> str:
    """
    Removes spaces from substrings in a Hebrew sentence that match the _HAGAR_RE regular expression.

    Parameters
    ----------
    heb_sen : str
        The input Hebrew sentence to process.

    Returns
    -------
    str
        The modified sentence with spaces removed from matched substrings.
    """
    return _HAGAR_RE.sub(lambda m: m.group().replace(' ', ''), heb_sen)


def correct_names(heb_line: str, eng_line: str) -> str:
    """
    Merge and correct Hebrew children's names in a line, guided by English tokens.
    This function standardizes specific Hebrew name variants based on cues from the corresponding English line.

    Parameters
    ----------
    heb_line : str
        The Hebrew text line containing names to be corrected.
    eng_line : str
        The corresponding English text line used to guide corrections.

    Returns
    -------
    str
        The Hebrew line with corrected and merged names."""
    out = correct_hagar(heb_line)
    if 'Leʔor' in eng_line and 'ל אור' in out:
        out = out.replace('ל אור', 'לאור')
    if 'Liʔōr' in eng_line and 'ל יאור' in out:
        out = out.replace('ל יאור', 'ליאור')
    return out

# word/phrase maps (subset; extend as needed)
MANUAL_JOIN: Dict[str, str] = {
    "ל כאן": "לכאן",
    "ל אן": "לאן",
    "שאת": "ש את",
    "ב בקשה": "בבקשה",
    "בבבקשה": "בבקשה",
    "וב": "ו ב",
    "ה יום": "היום",
    "ב סדר": "בסדר",
    "וה ": "ו ה ",
    " ול ": " ו ל ",
    "ואז": "ו אז",
    "בזה": "ב זה",
    "באמת": "ב אמת",
    "ה כול": "הכול",
    "ה כל": "הכול",
    "ל איבוד": "לאיבוד",
    "ב עצמך": " בעצמך",
    "מ יתוך": "מיתוך",
    "שכמו": "ש כמו",
    "ב חזרה": "בחזרה",
    "מ ימיה": "מימיה",
    "לאז": "ל אז",
    "מקור": "מ קור",
    "חדר אוכל": "חדראוכל",
    "כ אלה": "כאלה",
    "כ ואב": "כואב",
    "ל עצמה": "לעצמה",
    "ל עצמו": "לעצמו",
    'הא זו': 'הזו',
    'ל בריאות': 'לבריאות',
}

MWE_UNDERSCORE: Dict[str, str] = {
    "עוד פעם": "עוד_פעם",
    "כול הכבוד": "כול_ה_כבוד",
    "ה זה": "ה_זה",
    "איזה יופי": "איזה_יופי",
    "איך ש": "איך_ש",
    "עוד לא": "עוד_לא",
    "ה זאת": "ה_זאת",
    "תודה רבה": "תודה_רבה",
    "נכון מאוד": "נכון_מאוד",
    "אי אפשר": "אי_אפשר",
    "מה זאת אומרת": "מה_זאת_אומרת",
    "כול הכבוד": "כול_ה_כבוד",
}

UNK_LATIN_TO_HEB: Dict[str, str] = {
    "ʝirafa": "גירפה",
    "ʝinʝit": "גינגית",
    "ʝinʝi": "גינגי",
    "ʝulim": "גולים",
    "qoteʝ": "קוטג",
    "piʝama": "פיגמה",
    "ʝungel": "גונגל",
    "ʝimbori": "גימבורי",
    "ʝuqim": "גוקים",
    "ʝuq": "גונגל",
    "ʝirafot": "גירפות",
    "ʝins": "גינס",
    "ʝiraf": "גירף",
    "piʝamama": "פיגמה",
}

HEB_PUNCT_CHARS = ".,!?;:״׳\"”’—–-"

# used in align_punct_by_index
_INDEXABLE_PUNCT: set = set(list(".,!?;:"))

ENG_HEB_PREPOSITION: Dict[str, str] = {
    'ha#': 'ה', 'še#': 'ש', 'me#': 'מ', 'ma#': 'מ',
    'be#': 'ב', 'CLITIC': 'ה', 'ba#': 'ב',
    'la#': 'ל', 'le#': 'ל', 'we#': 'ו', 'bi#': 'ב',
    'ka#': 'כ', 'ke#': 'כ', 'kše#': 'כש', 'li#': 'ל',
    'mi#': 'מ', 'ta#': 'ת', 'u#': 'ו', 'wa#': 'ו',
}

# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def print_heb(text: str) -> None:
    """Naive RTL console print by reversing the string."""
    print(text[::-1])


def spaced_punct(text: str) -> str:
    """
    Adds spaces around specified Hebrew punctuation characters in the input text.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        String with spaces around punctuation and normalized whitespace.
    """
    text = re.sub(rf"([\{re.escape(HEB_PUNCT_CHARS)}])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def fix_gimel_apostrophe(heb: str, eng: str) -> str:
    if ' ʝ' in eng:
        heb = heb.replace(' ʝ', ' ג').replace('ʝ', 'ג')
    else:
        heb = heb.replace('ʝ', 'ג').replace(' ג', 'ג')
    return heb

def separate_common_clitics(text: str) -> str:
    # Separate Hebrew prefixes when glued: ו/ב/ל/כ/מ/ה before a Hebrew letter
    return re.sub(r"\b([ובלכמהה])(?=[\u0590-\u05FF])", r"\1 ", text)


def remove_junk(text: str) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t != "ם")


def drop_angle_brackets(text: str) -> str:
    """Remove literal '<' and '>' characters; keep inner content."""
    return re.sub(r"[<>]", "", text)


def add_missing_terminal_punct(heb: str, eng: str) -> str:
    if heb.rstrip().endswith(tuple(".!?")):
        return heb
    if eng.rstrip().endswith("."):
        return heb.rstrip() + "."
    if eng.rstrip().endswith("!"):
        return heb.rstrip() + "!"
    if eng.rstrip().endswith("?"):
        return heb.rstrip() + "?"
    return heb


def align_punct_by_index(heb: str, eng: str) -> str:
    """
    Aligns single-character punctuation tokens (.,!?;:) in Hebrew to match their positions in English.

    Parameters
    ----------
    heb : str
        Tokenized Hebrew sentence.
    eng : str
        Tokenized English sentence.

    Returns
    -------
    str
        Hebrew sentence with punctuation aligned to English.
    """
    e = eng.split()
    h = heb.split()
    if len(h) - len(e) > 8:
        return heb
    for i, tok in enumerate(e):
        if tok in _INDEXABLE_PUNCT:
            if i < len(h):
                if h[i] != tok and h[i] not in _INDEXABLE_PUNCT:
                    h.insert(i, tok)
            else:
                h.append(tok)
    return " ".join(h)


def apply_dict_joiners(text: str, mapping: Dict[str, str]) -> str:
    for bad, good in mapping.items():
        text = text.replace(bad, good)
    return text


def replace_unk_from_latin(heb: str, eng: str) -> str:
    h_toks = heb.split()
    e_toks = eng.split()
    if h_toks.count("[UNK]") == 0:
        return heb
    if len(h_toks) == len(e_toks):
        for i, (h, e) in enumerate(zip(h_toks, e_toks)):
            if h == "[UNK]":
                cand = UNK_LATIN_TO_HEB.get(e.lower())
                if cand:
                    h_toks[i] = cand
        return " ".join(h_toks)
    # lengths differ: replace the first UNK by best-effort if any aligned latin exists
    idx = h_toks.index("[UNK]")
    if idx < len(e_toks):
        cand = UNK_LATIN_TO_HEB.get(e_toks[idx].lower())
        if cand:
            h_toks[idx] = cand
    return " ".join(h_toks)


def add_clitic(heb_sen: str, eng_sen: str) -> str:
    """If ENG contains literal token 'CLITIC', insert Hebrew 'ה' at that index.
    This runs at the very end so Hebrew token indices are stable.
    """
    e = eng_sen.split()
    if 'CLITIC' not in e:
        return heb_sen
    i = e.index('CLITIC')
    h = heb_sen.split()
    i = max(0, min(i, len(h)))
    h.insert(i, 'ה')
    return ' '.join(h)


def correct_ha_ha(heb: str, eng: str) -> str:
    """
    Heuristic fixer for the special bigram case 'ha# ha#' (ENG) ↔ 'ה ה' (HEB).

    Usage with apply_if_improves:
        heb = apply_if_improves(heb, lambda s: correct_ha_ha(s, eng), E)

    If English contains the contiguous bigram 'ha# ha#' at some indices and the
    Hebrew tokens contain 'ה ה' at the same indices, we normalize by rejoining
    the Hebrew token list with single spaces (no token content change).
    Otherwise this is a no-op and returns the original `heb`.
    """
    # Split while preserving the bigram as a single chunk using capturing groups
    eng_list = [w for w in re.split(r'(\s+)|(ha#\s+ha#)', eng) if w and not w.isspace()]
    heb_list = [w for w in re.split(r'(\s+)|(ה\s+ה)', heb) if w and not w.isspace()]

    eng_indices = [i for i, x in enumerate(eng_list) if x == 'ha# ha#']
    heb_indices = [i for i, x in enumerate(heb_list) if x == 'ה ה']

    if eng_indices and eng_indices == heb_indices:
        return " ".join(heb_list)
    return heb

def _locate(seq: List[str], pred) -> List[int]:
    """Lightweight replacement for more_itertools.locate."""
    return [i for i, x in enumerate(seq) if pred(x)]

def correct_prepositions(heb: str, eng: str) -> str:
    """
    Index-aware fixups for Hebrew prepositions/clitics based on aligned English tokens.
    Implements the full mapping + logic from the spec.

    Notes
    -----
    - Safe on index errors: returns the original string on out-of-range.
    """
    heb_words = heb.split()
    eng_words = eng.split()

    keys = set(ENG_HEB_PREPOSITION.keys())

    try:
        indexes = _locate(eng_words, lambda x: x in keys)
        for i in indexes:
            eng_token = eng_words[i]
            heb_token = ENG_HEB_PREPOSITION[eng_token]

            # Already perfect at this position
            if i < len(heb_words) and heb_words[i] == heb_token:
                continue

            if i < len(heb_words) and heb_words[i].startswith(heb_token):
                # If next ENG token begins with the same clitic marker (e.g., 'ha# ...'),
                # insert the clitic without stripping the current token.
                if i + 1 < len(eng_words) and eng_words[i + 1].startswith(eng_token):
                    heb_words.insert(i, heb_token)
                else:
                    # Remove the clitic letter from current token and insert it as separate token
                    if heb_words[i].startswith(heb_token) and len(heb_words[i]) > 1:
                        heb_words[i] = heb_words[i][1:]
                    heb_words.insert(i, heb_token)
            else:
                # Not present at this position — insert explicitly
                heb_words.insert(i, heb_token)
    except Exception:
        return heb

    return ' '.join(heb_words)


# ---------------------------------------------------------------------------
# Alignment helpers & MOR parsing
# ---------------------------------------------------------------------------

def token_count(s: str) -> int:
    return len([t for t in s.split() if t])


def apply_if_improves(base: str, step_fn: Callable[[str], str], target_len: int) -> str:
    """
    Apply a transformation if it does not worsen the token-count distance to `target_len`.

    Parameters
    ----------
    base : str
        Original string.
    step_fn : Callable[[str], str]
        Transformation function.
    target_len : int
        Target token count.

    Returns
    -------
    str
        Transformed string if token-count distance does not worsen; else original.
    """
    before = token_count(base)
    after_s = step_fn(base)
    after = token_count(after_s)
    return after_s if abs(after - target_len) <= abs(before - target_len) else base
    return base

def parse_mor_tier(mor: Optional[str], expected_len: int) -> List[Optional[str]]:
    """Split a MOR tier string into per-token items, padded/truncated to `expected_len`."""
    if not mor:
        return [None] * expected_len
    items = mor.strip().split()
    if len(items) < expected_len:
        items += [None] * (expected_len - len(items))
    elif len(items) > expected_len:
        items = items[:expected_len]
    return items

def parse_mor_item(item: Optional[str]) -> Dict[str, str]:
    """Very light MOR parser: returns {pos, lemma, ...features}."""
    if not item:
        return {}
    pos = None
    rest = item
    if '|' in item:
        pos, rest = item.split('|', 1)
    feats: Dict[str, str] = {}
    parts = rest.split('&') if rest else []
    if parts:
        first = parts[0]
        if '=' not in first:
            feats['lemma'] = first
            parts = parts[1:]
    if pos:
        feats['pos'] = pos
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            feats[k] = v
        else:
            feats[p] = 'true'
    return feats

# ---------------------------------------------------------------------------
# Record & core processing
# ---------------------------------------------------------------------------
@dataclass
class TokenAlignRecord:
    eng_file_path: str
    heb_file_path: str
    utt_index: int
    speaker: str
    english: str
    hebrew_raw: str
    hebrew_segmented: str
    mor: Optional[str]
    eng_len: int
    heb_len: int
    aligned: bool


def process_pair(english: str, hebrew: str) -> str:
    # 0) Preprocess (user regex-based spacing of tags/punct)
    heb = preprocess_heb_line(hebrew)
    eng = preprocess_eng_line(english)

    # 1) DictaBERT segmentation (or fallback)
    heb = segment_heb(heb)
    E = token_count(eng)

    # 2) Names → 2.5) drop < > → 3+) connections & normalizations
    heb = apply_if_improves(heb, lambda s: correct_names(s, eng), E)
    heb = apply_if_improves(heb, drop_angle_brackets, E)
    heb = apply_if_improves(heb, lambda s: apply_dict_joiners(s, MANUAL_JOIN), E)

    # 3) Normalizations and spacing heuristics
    heb = apply_if_improves(heb, lambda s: fix_gimel_apostrophe(s, eng), E)
    heb = apply_if_improves(heb, lambda s: replace_unk_from_latin(s, eng), E)
    heb = apply_if_improves(heb, separate_common_clitics, E)
    heb = apply_if_improves(heb, remove_junk, E)
    heb = apply_if_improves(heb, spaced_punct, E)
    heb = apply_if_improves(heb, lambda s: align_punct_by_index(s, eng), E)
    heb = add_missing_terminal_punct(heb, eng)
    heb = apply_if_improves(heb, lambda s: correct_prepositions(s, eng), E)
    heb = apply_if_improves(heb, lambda s: apply_dict_joiners(s, MWE_UNDERSCORE), E)
    heb = apply_if_improves(heb, lambda s: correct_ha_ha(s, eng), E)
    heb = apply_if_improves(heb, lambda s: correct_prepositions(s, eng), E)
    

    # 4) Add CLITIC after all connections (by English index)
    heb = apply_if_improves(heb, lambda s: add_clitic(s, eng), E)

    return re.sub(r"\s+", " ", heb).strip(), re.sub(r"\s+", " ", eng).strip()

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_token_level(
    *,
    in_pairs_jsonl: Path,
    out_pairs_token_jsonl: Path,
    out_tokens_jsonl: Path,
    out_hebrew_token_txt: Path,
    details_parquet: Path,
    remove_speakers: Tuple[str, ...] = ("CHI",),
    verbose: bool = True,
    progress_every: int = 1000,
) -> Tuple[Path, Path, Path]:
    out_pairs_token_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_hebrew_token_txt.parent.mkdir(parents=True, exist_ok=True)

    print("[token-level] START")
    print(f"[token-level] in_pairs_jsonl = {in_pairs_jsonl}")
    print(f"[token-level] out_pairs_token_jsonl = {out_pairs_token_jsonl}")
    print(f"[token-level] out_hebrew_token_txt = {out_hebrew_token_txt}")
    print(f"[token-level] details_parquet = {details_parquet}")
    print(f"[token-level] remove_speakers = {remove_speakers}")
    print(f"[token-level] segmenter = {'DictaBERT' if _SEGMENT_WITH_DICTABERT else 'fallback'}")

    rows: List[TokenAlignRecord] = []

    # Pre-count total input lines for an accurate progress bar
    try:
        total_lines = sum(1 for _ in open(in_pairs_jsonl, "r", encoding="utf-8"))
    except Exception:
        total_lines = None

    processed = 0
    aligned_count = 0
    skipped_count = 0
    misaligned_printed = 0
    token_rows_written = 0

    with open(in_pairs_jsonl, "r", encoding="utf-8") as fin, \
         open(out_pairs_token_jsonl, "w", encoding="utf-8") as fout_pairs, \
         open(out_tokens_jsonl, "w", encoding="utf-8") as fout_tokens, \
         open(out_hebrew_token_txt, "w", encoding="utf-8") as fout_txt:
        for line in tqdm(fin, total=total_lines, desc="[token-level] aligning", unit="utt"):
            rec = json.loads(line)
            speaker = (rec.get("speaker") or "").strip()
            if speaker in remove_speakers:
                skipped_count += 1
                continue  # CDS only (exclude child)

            eng = (rec.get("english") or "").strip()
            heb_raw = (rec.get("hebrew") or "").strip()
            mor = rec.get("mor")

            heb_seg, eng_seg = process_pair(eng, heb_raw)

            e_len = token_count(eng_seg)
            h_len = token_count(heb_seg)
            aligned = (e_len == h_len)
            aligned_count += int(aligned)
            processed += 1

            if verbose and progress_every and processed % progress_every == 0:
                ratio = aligned_count / max(processed, 1)
                print(f"[progress] processed={processed} aligned={aligned_count} ({ratio:.1%}) skipped={skipped_count}")

            if verbose and not aligned and misaligned_printed < 5:
            # if verbose and not aligned:
                print("[warn] misaligned example:")
                print(f"  ENG({e_len}): {eng_seg}")
                print(f"  HEB({h_len}): ")
                print_heb(f"{heb_seg}")
                misaligned_printed += 1

            out_obj = {
                "eng_file_path": rec.get("eng_file_path"),
                "heb_file_path": rec.get("heb_file_path"),
                "utt_index": rec.get("utt_index"),
                "speaker": speaker,
                "english": eng_seg,
                "hebrew_raw": heb_raw,
                "hebrew_segmented": heb_seg,
                "mor": mor,
                "eng_len": e_len,
                "heb_len": h_len,
                "aligned": aligned,
            }
            fout_pairs.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout_txt.write(heb_seg + "\n")

            # per-token rows with MOR annotation
            e_toks = eng_seg.split()
            h_toks = heb_seg.split()
            mor_items = parse_mor_tier(mor if isinstance(mor, str) else None, len(e_toks))
            for idx, e_tok in enumerate(e_toks):
                h_tok = h_toks[idx] if idx < len(h_toks) else None
                mor_tok = mor_items[idx] if idx < len(mor_items) else None
                mor_parsed = parse_mor_item(mor_tok)
                token_row = {
                    "eng_file_path": rec.get("eng_file_path"),
                    "heb_file_path": rec.get("heb_file_path"),
                    "utt_index": rec.get("utt_index"),
                    "speaker": speaker,
                    "eng_idx": idx,
                    "eng_token": e_tok,
                    "heb_token": h_tok,
                    "mor_token": mor_tok,
                    "mor_parsed": mor_parsed,  # {pos, lemma, num, gen, ... if present}
                    "aligned_utt": aligned,
                }
                fout_tokens.write(json.dumps(token_row, ensure_ascii=False) + "\n")
                token_rows_written += 1

            rows.append(
                TokenAlignRecord(
                    eng_file_path=str(rec.get("eng_file_path")),
                    heb_file_path=str(rec.get("heb_file_path")),
                    utt_index=int(rec.get("utt_index", -1)),
                    speaker=speaker,
                    english=eng,
                    hebrew_raw=heb_raw,
                    hebrew_segmented=heb_seg,
                    mor=mor if isinstance(mor, (str, type(None))) else str(mor),
                    eng_len=e_len,
                    heb_len=h_len,
                    aligned=aligned,
                )
            )

    # Diagnostics
    df = pd.DataFrame([
        {
            "eng_file_path": r.eng_file_path,
            "heb_file_path": r.heb_file_path,
            "utt_index": r.utt_index,
            "speaker": r.speaker,
            "eng_len": r.eng_len,
            "heb_len": r.heb_len,
            "aligned": r.aligned,
        }
        for r in rows
    ])
    df.to_parquet(details_parquet, index=False)

    total = processed + skipped_count
    final_ratio = (aligned_count / processed) if processed else 0.0
    print("[token-level] DONE")
    print(f"  total_input={total} processed={processed} skipped={skipped_count}")
    print(f"  aligned={aligned_count}/{processed} ({final_ratio:.1%})")
    print(f"  wrote pairs  -> {out_pairs_token_jsonl}")
    print(f"  wrote tokens  -> {out_tokens_jsonl} (rows={token_rows_written})")
    print(f"  wrote hebrew -> {out_hebrew_token_txt}")
    print(f"  wrote diag   -> {details_parquet}")
    print(f"  segmenter    -> {'DictaBERT' if _SEGMENT_WITH_DICTABERT else 'fallback'}")

    return out_pairs_token_jsonl, out_tokens_jsonl, out_hebrew_token_txt, details_parquet


def main(
    *,
    in_pairs_jsonl: Optional[Path] = None,
    out_pairs_token_jsonl: Optional[Path] = None,
    out_tokens_jsonl: Optional[Path] = None,
    out_hebrew_token_txt: Optional[Path] = None,
    details_parquet: Optional[Path] = None,
    remove_speakers: Tuple[str, ...] = ("CHI",),
):
    if in_pairs_jsonl is None:
        in_pairs_jsonl = configs.Dirs.processed / "htberman" / "aligned_pairs.jsonl"
    if out_pairs_token_jsonl is None:
        out_pairs_token_jsonl = configs.Dirs.processed / "htberman" / "aligned_pairs_token.jsonl"
    if out_tokens_jsonl is None:
        out_tokens_jsonl = configs.Dirs.processed / "htberman" / "aligned_tokens.jsonl"
    if out_hebrew_token_txt is None:
        out_hebrew_token_txt = configs.Dirs.processed / "htberman" / "corpus_utt_token_aligned.txt"
    if details_parquet is None:
        details_parquet = configs.Dirs.processed / "htberman" / "details_token.parquet"

    return run_token_level(
        in_pairs_jsonl=in_pairs_jsonl,
        out_pairs_token_jsonl=out_pairs_token_jsonl,
        out_tokens_jsonl=out_tokens_jsonl,
        out_hebrew_token_txt=out_hebrew_token_txt,
        details_parquet=details_parquet,
        remove_speakers=remove_speakers,
        verbose=False,
    )


if __name__ == "__main__":
    main()