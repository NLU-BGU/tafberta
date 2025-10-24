from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import argparse
import json
import re
import string
import pandas as pd
import unicodedata

from tafberta import configs


# ----------------------------
# %mor parsing + helpers
# ----------------------------
_NUM_ALLOWED = {"sg", "pl"}
_GEN_ALLOWED = {"ms", "fm"}

def _parse_mor(mor: Optional[Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse CHILDES %mor for one token; return (pos, num, gen).
    Assumes number∈{sg,pl}, gender∈{ms,fm} (anything else dropped).
    """
    if not mor:
        return None, None, None

    num: Optional[str] = None
    gen: Optional[str] = None
    pos: Optional[str] = None

    if isinstance(mor, dict):
        raw_pos = mor.get("pos")
        if isinstance(raw_pos, str):
            pos = raw_pos.strip().lower() or None

        for key, value in mor.items():
            if isinstance(key, str):
                key_lower = key.strip().lower()

                if num is None:
                    if key_lower == "num" and isinstance(value, str):
                        candidate = value.strip().lower()
                        if candidate in _NUM_ALLOWED:
                            num = candidate
                    elif key_lower.startswith("num:"):
                        candidate = key_lower.split(":", 1)[1]
                        if candidate in _NUM_ALLOWED:
                            num = candidate

                if gen is None:
                    if key_lower == "gen" and isinstance(value, str):
                        candidate = value.strip().lower()
                        if candidate in _GEN_ALLOWED:
                            gen = candidate
                    elif key_lower.startswith("gen:"):
                        candidate = key_lower.split(":", 1)[1]
                        if candidate in _GEN_ALLOWED:
                            gen = candidate

            if pos is not None and num is not None and gen is not None:
                break

        return pos, num, gen

    if not isinstance(mor, str):
        return None, None, None

    text = mor.strip()
    if not text:
        return None, None, None

    right = text
    if "|" in text:
        pos_part, right = text.split("|", 1)
        pos = pos_part.strip().lower() or None
    else:
        tokens = text.split(None, 1)
        if tokens:
            pos = tokens[0].strip().lower() or None
            right = tokens[1] if len(tokens) > 1 else ""
        else:
            return None, None, None

    feature_blob = right.split("=", 1)[0]
    for segment in feature_blob.split("&"):
        if ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        key = key.strip().lower()
        value = value.strip().lower()

        if num is None and key == "num" and value in _NUM_ALLOWED:
            num = value
        elif gen is None and key == "gen" and value in _GEN_ALLOWED:
            gen = value

        if num is not None and gen is not None:
            break

    if (num is None or gen is None) and feature_blob:
        for token in re.split(r"[\s:/,;=-]+", feature_blob.lower()):
            token = token.strip()
            if not token:
                continue
            if num is None and token in _NUM_ALLOWED:
                num = token
            elif gen is None and token in _GEN_ALLOWED:
                gen = token
            if num is not None and gen is not None:
                break

    return pos, num, gen

def _is_noun(pos: Optional[str]) -> bool:
    return pos and pos == "n"

def _safe_get(d: Dict, keys: Iterable[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

# ----------------------------
# Entry container
# ----------------------------
@dataclass
class NounEntry:
    heb_token: str
    eng_word: str
    num: str  # "", "sg", "pl"
    gen: str  # "", "ms", "fm"

# ----------------------------
# Your post-filters (list-level)
# ----------------------------
def _get_heb_token(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("heb_token", ""))
    if hasattr(obj, "heb_token"):
        return str(getattr(obj, "heb_token"))
    if hasattr(obj, "get"):
        try:
            return str(obj.get("heb_token", ""))
        except Exception:
            return ""
    return ""

def remove_single_char_strings(strings: List[Any]):
    return [s for s in strings if len(_get_heb_token(s)) > 1]

def remove_strings_with_punctuation(strings: List[Any]):
    punctuation_set = set(string.punctuation)
    return [s for s in strings
            if not any(ch in punctuation_set for ch in _get_heb_token(s))
            ]

# ----------------------------
# Sound mapping + filter
# ----------------------------

def map_sounds() -> Dict[str, List[str]]:
    """
    Map *leading* Latin transliteration graphemes to plausible Hebrew onsets.

    Notes:
    - Includes both precomposed (e.g., 'š', 'ṭ', 'ṣ', 'ḥ', 'ḳ', 'ō', 'ū', 'ī', 'ā', 'ē')
      and combining-diacritic forms (e.g., 'š', 'ṭ', 'ṣ', 'ḥ', 'ḳ', 'ō', 'ū', 'ī', 'ā', 'ē').
    - Keeps your earlier simple mappings (a/e/i/o/u …, sh/ch/ts …) as fallbacks.
    - ʕ (ayin) → 'ע'; ʔ (glottal) → 'א' (sometimes also matches 'ע').
    - q and ḳ/ ḳ → 'ק'; ḥ/ ḥ/ kh/ x → 'ח'/'כ'.
    """
    # Base (your previous) set
    base = {
        'a': ['א', 'ע'],
        'e': ['א', 'ע'],
        'i': ['י', 'א'],
        'o': ['ו', 'ע', 'א'],  # include ו for /o/
        'u': ['ו', 'א'],
        'b': ['ב'],
        'c': ['כ', 'ק', 'צ'],
        'k': ['כ', 'ק'],
        'q': ['ק'],
        'd': ['ד'],
        'f': ['פ', 'ף'],
        'g': ['ג'],
        'h': ['ה', 'ח'],
        'l': ['ל'],
        'm': ['מ', 'ם'],
        'n': ['נ', 'ן'],
        'p': ['פ', 'ף'],
        'r': ['ר'],
        's': ['ס', 'ש'],   # fallback if just 's'
        't': ['ת', 'ט'],   # fallback if just 't'
        'z': ['ז'],
        # digraphs
        'ch': ['ח', 'כ'],
        'sh': ['ש'],
        'th': ['ת'],
        'ts': ['צ'],
        'kh': ['ח', 'כ'],
        'ye': ['י'],
        'xa': ['ח'],
        'ya': ['י'],
        # j seen in your UNK fixes (e.g., jirafa → גירפה)
        'j': ['ג', "ג׳"],
        'u#': ['ו'],  # keep some legacy shorthands if they appear at start
        'wa#': ['ו'],
        'we#': ['ו'],
        'bi#': ['ב'],
        'ba#': ['ב'],
        'be#': ['ב'],
        'la#': ['ל'],
        'le#': ['ל'],
        'li#': ['ל'],
        'ma#': ['מ'],
        'mi#': ['מ'],
        'ka#': ['כ'],
        'ke#': ['כ'],
        'ta#': ['ת'],
        'kše#': ['כש'],   # just in case such prefixes leak into tokens
    }

    # Diacritic / extended graphemes (both precomposed and combining)
    extended = {
        # š / š  → ש
        'š': ['ש'], 'š': ['ש'],
        # ṭ / ṭ  → ט
        'ṭ': ['ט'], 'ṭ': ['ט'],
        # ṣ / ṣ  → צ
        'ṣ': ['צ'], 'ṣ': ['צ'],
        # ḥ / ḥ  → ח
        'ḥ': ['ח'], 'ḥ': ['ח'],
        # ḳ / ḳ  → ק
        'ḳ': ['ק'], 'ḳ': ['ק'],
        # q already present; include ḳ alt above

        # Vowels with macrons (ā ē ī ō ū), plus combining macron forms
        'ā': ['א', 'ע'], 'ā': ['א', 'ע'],
        'ē': ['א', 'ע'], 'ē': ['א', 'ע'],
        'ī': ['י', 'א'], 'ī': ['י', 'א'],
        'ō': ['ו', 'ע', 'א'], 'ō': ['ו', 'ע', 'א'],
        'ū': ['ו', 'א'], 'ū': ['ו', 'א'],

        # Glottal + pharyngeal
        'ʔ': ['א', 'ע'],  # alef (sometimes ayin in transcriptions)
        'ʕ': ['ע'],
    }

    # Merge (extended wins for specific keys)
    base.update(extended)
    return base


def _leading_grapheme(s: str, keys: List[str]) -> Optional[str]:
    """
    Return the **longest** mapping key that matches the *start* of `s`.
    Handles digraphs (sh, ts, kh) and diacritic sequences (š, ṭ, ō …).
    """
    if not s:
        return None

    # Normalize just enough to keep combining marks intact but standardize compatibility chars
    s_norm = unicodedata.normalize("NFKC", s)

    # Try longest keys first (up to, say, 3–4 chars covers š / ṭ / ō and digraphs)
    for L in range(min(4, len(s_norm)), 0, -1):
        cand = s_norm[:L]
        if cand in keys:
            return cand
    return None


def filter_hebrew_words_by_sound(nouns_dict):
    """
    Improved selector:
      • detects a leading *grapheme* using longest-prefix match against map keys
      • supports diacritic forms (š/ š, ṭ/ ṭ, ṣ/ ṣ, ō/ ō, etc.)
      • if the word begins with ʔ and we don't match it, we skip it once and retry
    """
    sound_map = map_sounds()
    all_keys = list(sound_map.keys())
    filtered_dict = {}

    for eng_word, entries in nouns_dict.items():
        if not eng_word:
            filtered_dict[eng_word] = []
            continue

        ew = eng_word.strip().lower()
        if not ew:
            filtered_dict[eng_word] = []
            continue

        # Try full grapheme first
        key = _leading_grapheme(ew, all_keys)

        # If nothing matched and we have a leading glottal ʔ, skip it once
        if key is None and ew and ew[0] == 'ʔ':
            ew2 = ew[1:]
            key = _leading_grapheme(ew2, all_keys)

        # Fallbacks: 2-letter digraphs, then single char
        if key is None and len(ew) >= 2 and ew[:2] in sound_map:
            key = ew[:2]
        if key is None:
            key = ew[0]

        sound_matches = sound_map.get(key, sound_map.get(key[:1], []))

        filtered_entries = [
            entry for entry in entries
            if any(entry.heb_token.startswith(h_onset) for h_onset in sound_matches)
        ]
        filtered_dict[eng_word] = filtered_entries

    return filtered_dict

# def map_sounds():
#     # Expanded mapping between English sounds and Hebrew letters
#     return {
#         'a': ['א', 'ע'],
#         'e': ['א', 'ע'],
#         'i': ['י', 'א'],
#         'o': ['ע', 'א'],
#         'u': ['ו', 'א'],
#         'b': ['ב'],
#         'c': ['כ', 'ק', 'צ'],
#         'k': ['כ', 'ק'],
#         'd': ['ד'],
#         'f': ['פ', 'ף'],
#         'g': ['ג'],
#         'h': ['ה', 'ח'],
#         'l': ['ל'],
#         'm': ['מ', 'ם'],
#         'n': ['נ', 'ן'],
#         'p': ['פ', 'ף'],
#         'r': ['ר'],
#         's': ['ס', 'ש'],
#         't': ['ת', 'ט'],
#         'z': ['ז'],
#         'ch': ['ח', 'כ'],
#         'sh': ['ש'],
#         'th': ['ת'],
#         'ts': ['צ'],
#         'ye': ['י'],
#         'xa': ['ח'],
#         'ya': ['h'],
#     }

# def filter_hebrew_words_by_sound(nouns_dict: Dict[str, List[NounEntry]]):
#     """
#     Apply your sound-based filtering *per eng_word*.
#     NOTE: As in your sketch, we primarily look at the first Latin char,
#     with a small guard for leading ʔ. We also try a 2-letter prefix first
#     when available to honor 'sh','ch','ts','th','ye','xa','ya'.
#     """
#     # TODO: expand sound map with grapheme like š
#     sound_map = map_sounds()
#     filtered_dict: Dict[str, List[NounEntry]] = {}

#     for eng_word, entries in nouns_dict.items():
#         if not eng_word:
#             filtered_dict[eng_word] = []
#             continue

#         ew = eng_word.strip().lower()
#         if not ew:
#             filtered_dict[eng_word] = []
#             continue

#         # Your logic: handle a leading glottal ʔ by skipping it
#         if ew[0] == 'ʔ' and len(ew) > 1:
#             ew = ew[1:]

#         # Prefer 2-letter digraph if present in map; otherwise fallback to first char
#         prefix2 = ew[:2]
#         if prefix2 in sound_map:
#             sound_matches = sound_map[prefix2]
#         else:
#             sound_matches = sound_map.get(ew[0], [])

#         filtered_entries = [
#             entry for entry in entries
#             if any(entry.heb_token.startswith(sound) for sound in sound_matches)
#         ]
#         filtered_dict[eng_word] = filtered_entries

#     return filtered_dict

# ----------------------------
# Main filtering routine
# ----------------------------
def filter_htberman_nouns_to_df(
    jsonl_path: Path,
    *,
    save_csv: Optional[Path] = None,
    save_parquet: Optional[Path] = None,
) -> pd.DataFrame:
    """
    From per-token JSONL (aligned_tokens.jsonl), extract nouns, then:
      1) build nouns_dict[eng_word] = list[NounEntry]
      2) nouns_dict[eng] = remove_punct(remove_single_char(...))
      3) nouns_dict_2fix = groups with len > 1
      4) filtered_dict = filter_hebrew_words_by_sound(nouns_dict_2fix)
      5) filtered_dict_fixed = keep only groups where len == 1
      6) final = union of:
            - groups that were already unique after step (2)
            - the newly unique groups from (5)
    Return a DataFrame: heb_token, eng_word, num, gen
    """
    # ---- collect noun candidates
    nouns: List[NounEntry] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = json.loads(ln)

            mor = rec.get("mor_parsed")
            pos, num, gen = _parse_mor(mor)
            if not _is_noun(pos):
                continue

            heb_token = _safe_get(rec, ("heb_token", "hebrew_token", "hebrew"), "") or ""
            eng_word  = _safe_get(rec, ("eng_word", "eng_token", "english_token", "english"), "") or ""

            nouns.append(
                NounEntry(
                    heb_token=heb_token,
                    eng_word=eng_word,
                    num=num or "",
                    gen=gen or "",
                )
            )

    # ---- step 1 + 2: group → clean per-eng_word
    nouns_dict: Dict[str, List[NounEntry]] = {}
    for entry in nouns:
        nouns_dict.setdefault(entry.eng_word, []).append(entry)
    # TODO: after the previous for loop is done, iterate over nouns_dict values such that the output of the for loop is nouns_dict where each value is a list of unique entries
    for eng_word, entries in nouns_dict.items():
        seen_keys = set()
        unique_entries: List[NounEntry] = []
        for entry in entries:
            entry_key = (entry.heb_token, entry.num, entry.gen)
            if entry_key in seen_keys:
                continue
            seen_keys.add(entry_key)
            unique_entries.append(entry)
        nouns_dict[eng_word] = unique_entries

    # apply remove_* filters for each group
    for eng, entries in list(nouns_dict.items()):
        nouns_dict[eng] = remove_strings_with_punctuation(remove_single_char_strings(entries))

    # ---- step 3: groups with > 1 (need disambiguation)
    nouns_dict_2fix = {eng: vals for eng, vals in nouns_dict.items() if len(vals) > 1}

    # ---- step 4 + 5: sound-based filter → keep only newly unique
    filtered_dict = filter_hebrew_words_by_sound(nouns_dict_2fix)
    filtered_dict_fixed = {eng: vals for eng, vals in filtered_dict.items() if len(vals) == 1}

    # ---- step 6: merge results
    final_rows: List[NounEntry] = []

    # (a) groups already unique after initial cleaning
    for eng, vals in nouns_dict.items():
        if len(vals) == 1:
            final_rows.append(vals[0])

    # (b) groups newly unique after sound filtering (avoid duplicates)
    seen_keys = {(r.eng_word, r.heb_token) for r in final_rows}
    for eng, vals in filtered_dict_fixed.items():
        if not vals:
            continue
        cand = vals[0]
        key = (cand.eng_word, cand.heb_token)
        if key not in seen_keys:
            final_rows.append(cand)
            seen_keys.add(key)

    # -> DataFrame
    df = pd.DataFrame(
        [
            {
                "heb_token": r.heb_token,
                "eng_word": r.eng_word,
                "num": r.num,
                "gen": r.gen,
            }
            for r in final_rows
        ],
        columns=["heb_token", "eng_word", "num", "gen"],
    )

    if save_csv:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False, encoding="utf-8")
    if save_parquet:
        save_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_parquet, index=False)

    return df


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Fix HTBerman nouns by grouping + sound filtering")
    p.add_argument("--in", dest="inp", type=Path,
                   default=configs.Dirs.processed / "htberman" / "aligned_tokens.jsonl",
                   help="Input per-token JSONL")
    p.add_argument("--out-csv", dest="out_csv", type=Path,
                   default=configs.Dirs.heclimp_legal_words / 'htberman_nouns.csv',
                   help="Optional CSV output path")
    p.add_argument("--out-parquet", dest="out_parquet", type=Path, default=None,
                   help="Optional Parquet output path")
    args = p.parse_args()

    print("Filtering HTBerman nouns from:", args.inp)

    out_df = filter_htberman_nouns_to_df(args.inp, save_csv=args.out_csv, save_parquet=args.out_parquet)
    print(f"Resulting nouns: {len(out_df)}")
    if args.out_csv:
        print(f"on {args.out_csv}")
    elif args.out_parquet:
        print(f"on {args.out_parquet}")
