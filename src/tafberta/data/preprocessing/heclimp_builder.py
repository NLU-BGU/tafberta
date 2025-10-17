# src/tafberta/data/preprocessing/heclimp_builder.py
from pathlib import Path
from typing import Optional, Tuple

import argparse
import pandas as pd

from tafberta import configs


# ----------------------------
# IO helpers
# ----------------------------

def _default_nouns_path() -> Path:
    """Return default location for nouns (produced by heclimp_prepare)."""
    heclimp_csv = configs.Dirs.processed / "heclimp" / "nouns.csv"
    return heclimp_csv


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def get_nouns(nouns_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load nouns table with columns: ['heb_token','eng_word','num','gen'].
    Assumes num∈{sg,pl}, gen∈{ms,fm}.
    """
    if nouns_path is not None:
        if not nouns_path.exists():
            raise FileNotFoundError(f"nouns file not found: {nouns_path}")
        df = _read_table(nouns_path)
    else:
        path = _default_nouns_path()
        if path.exists():
            df = _read_table(path)
        else:
            raise FileNotFoundError(
                "Could not find nouns table. "
                f"Tried {path}. Provide --nouns."
            )

    need = {"heb_token", "eng_word", "num", "gen"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"nouns table missing columns: {missing}")

    df = df.copy()
    for c in ("heb_token", "eng_word", "num", "gen"):
        df[c] = df[c].astype(str)
    return df


# ----------------------------
# Demonstratives inventory
# ----------------------------

def get_demonstratives() -> pd.DataFrame:
    """
    Hebrew demonstratives (near/far, sg/pl, ms/fm).
    """
    demonstratives = {
        "word":       ["זו", "זה", "אלה", "אלו", "היא", "הוא", "הן", "הם"],
        "num":        ["sg", "sg", "pl", "pl", "sg", "sg", "pl", "pl"],
        "gen":        ["fm", "ms", "fm", "ms", "fm", "ms", "fm", "ms"],
        "proximity":  ["near", "near", "near", "near", "far", "far", "far", "far"],
    }
    return pd.DataFrame(demonstratives)


def get_unique_vals_each_col(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise unique values (convenience table)."""
    unique_values = {col: df[col].unique() for col in df.columns}
    return pd.DataFrame({k: pd.Series(v) for k, v in unique_values.items()})


def _build_demonstratives_views(df_demonstratives: pd.DataFrame):
    """Return (df_demonstratives_indexed, unique_df) for quick lookup."""
    df_idx = df_demonstratives.set_index("word", drop=True).copy()
    unique_df = get_unique_vals_each_col(df_demonstratives[["num", "gen", "proximity"]])
    return df_idx, unique_df


def _get_opposite_demonstrative(
    curr_word: str,
    df_demonstratives_idx: pd.DataFrame,
    unique_demon_df: pd.DataFrame,
    chang_var: str = "gen",
) -> str:
    """
    Given a demonstrative 'curr_word', flip either its 'gen' or 'num' while
    keeping the other features the same (including proximity), and return
    the *other* demonstrative word.
    """
    curr = df_demonstratives_idx.loc[curr_word]
    other_cols = [c for c in df_demonstratives_idx.columns if c != chang_var]

    # opposite value for the change variable
    oppos_value = unique_demon_df[unique_demon_df[chang_var] != curr[chang_var]][chang_var].iloc[0]

    # same others, opposite this var
    mask_same_others = (df_demonstratives_idx[other_cols] == curr[other_cols]).all(axis=1)
    mask_opposite    = df_demonstratives_idx[chang_var] == oppos_value
    opp_df = df_demonstratives_idx[mask_same_others & mask_opposite]
    if opp_df.empty:
        return curr_word  # shouldn't happen with our inventory
    return opp_df.index[0]


def _return_noun_demonstrative(
    row: pd.Series,
    *,
    df_demonstratives_idx: pd.DataFrame,
    unique_demon_df: pd.DataFrame,
) -> pd.Series:
    """
    For a merged noun row (has columns: heb_token, num, gen, word, proximity),
    return right demonstrative and the wrong ones for num/gen.
    In plural ('אלה'/'אלו') the 'wrong_gen' is not meaningful → return None.
    """
    right = row["word"]
    res = {
        "heb_token": row["heb_token"],
        "right_demonstrative": right,
        "opposite_demonstrative_num": _get_opposite_demonstrative(
            right, df_demonstratives_idx, unique_demon_df, chang_var="num"
        ),
        "opposite_demonstrative_gen": None,
    }

    # Only produce a gender-flipped demonstrative if the base form encodes gender
    if right not in {"אלה", "אלו"}:
        res["opposite_demonstrative_gen"] = _get_opposite_demonstrative(
            right, df_demonstratives_idx, unique_demon_df, chang_var="gen"
        )
    return pd.Series(res)


def create_df_nouns_right_wrong_demonstratives(nouns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge nouns with demonstratives by (num, gen), then compute:
      - right_demonstrative
      - opposite_demonstrative_num
      - opposite_demonstrative_gen (None for 'אלה'/'אלו')
    """
    df_demonstratives = get_demonstratives()
    merged = pd.merge(nouns_df, df_demonstratives, on=["num", "gen"], how="inner")
    df_demonstratives_idx, unique_demon_df = _build_demonstratives_views(df_demonstratives)

    out_cols = ["heb_token", "right_demonstrative", "opposite_demonstrative_num", "opposite_demonstrative_gen"]
    merged[out_cols] = merged.apply(
        _return_noun_demonstrative,
        axis=1,
        result_type="expand",
        df_demonstratives_idx=df_demonstratives_idx,
        unique_demon_df=unique_demon_df,
    )
    return merged


# ----------------------------
# Sentence writer → TWO FOLDERS
# ----------------------------

def write_minimal_pairs_dirs(
    df_pairs: pd.DataFrame,
    outroot: Path,
    paradigm: str = "agreement_determiner_noun-across_0_adjective",
    template: str = "תסתכל על ה {} ה{} .",
) -> Tuple[Path, Path]:
    """
    Write two sibling folders:
      {paradigm}_num/all_data.txt  and  {paradigm}_gen/all_data.txt

    Each file alternates: odd line = WRONG, even line = RIGHT.
    """
    out_num_dir = outroot / f"{paradigm}_num"
    out_gen_dir = outroot / f"{paradigm}_gen"
    out_num_dir.mkdir(parents=True, exist_ok=True)
    out_gen_dir.mkdir(parents=True, exist_ok=True)

    f_num = out_num_dir / "all_data.txt"
    f_gen = out_gen_dir / "all_data.txt"

    with open(f_num, "w", encoding="utf-8") as num_out, open(f_gen, "w", encoding="utf-8") as gen_out:
        for _, row in df_pairs.iterrows():
            noun = str(row["heb_token"]).strip()
            dem_right = str(row["right_demonstrative"]).strip()
            dem_wrong_num = str(row["opposite_demonstrative_num"]).strip()

            # number contrast (always present)
            bad_num = template.format(noun, dem_wrong_num)
            good   = template.format(noun, dem_right)
            num_out.write(bad_num + "\n")
            num_out.write(good + "\n")

            # gender contrast (only if applicable)
            dem_wrong_gen = row.get("opposite_demonstrative_gen", None)
            if isinstance(dem_wrong_gen, str) and dem_wrong_gen.strip():
                bad_gen = template.format(noun, str(dem_wrong_gen).strip())
                gen_out.write(bad_gen + "\n")
                gen_out.write(good + "\n")

    return f_gen, f_num


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HeCLiMP minimal pairs from HTBerman nouns (two-folder layout)")
    p.add_argument("--nouns", type=Path,
                   default=configs.Dirs.heclimp_legal_words / 'htberman_nouns.csv',
                   help="Path to nouns table (CSV/Parquet) with columns heb_token,eng_word,num,gen. "
                        "Defaults to processed/heclimp/nouns.csv.")
    p.add_argument("--outroot", type=Path,
                   default=configs.Dirs.heclimp_testsuits_htberman,
                   help="Root output directory. Two subfolders will be created under it.")
    p.add_argument("--paradigm", type=str,
                   default="agreement_determiner_noun-across_0_adjective",
                   help="Paradigm name used as folder prefix")
    p.add_argument("--template", type=str,
                   default="תסתכל על ה {} ה{} .",
                   help="Sentence template; use two '{}' slots: noun, demonstrative")
    return p.parse_args()


def _default_outroot() -> Path:
    # Prefer a testsuites dir if present; else fall back under processed/heclimp
    try:
        return configs.Dirs.heclimp_testsuits_htberman
    except Exception:
        return (configs.Dirs.processed / "heclimp" / "sentences")


def main() -> None:
    args = parse_args()

    outroot = args.outroot or _default_outroot()
    nouns_df = get_nouns(args.nouns)

    print(f"[heclimp_builder] nouns loaded: {len(nouns_df)} rows")
    pairs_df = create_df_nouns_right_wrong_demonstratives(nouns_df)
    print(f"[heclimp_builder] pairs ready: {len(pairs_df)} rows")

    f_gen, f_num = write_minimal_pairs_dirs(
        pairs_df,
        outroot=outroot,
        paradigm=args.paradigm,
        template=args.template,
    )
    print("[heclimp_builder] wrote:")
    print("  GEN →", f_gen)
    print("  NUM →", f_num)


if __name__ == "__main__":
    main()
