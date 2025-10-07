from pathlib import Path
from typing import Dict, Iterable
import os, shutil

from tafberta import configs

def copy_new_files(src_dir: Path, dst_dir: Path) -> None:
    """
    Copy files from src_dir to dst_dir only if they don't already exist.
    Creates dst_dir if it does not exist.

    Parameters
    ----------
    src_dir : str
        Source directory path.
    dst_dir : str
        Destination directory path.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fn in os.listdir(src_dir):
        sp, dp = src_dir / fn, dst_dir / fn
        if sp.is_file():
            if not dp.exists():
                shutil.copy2(sp, dp); print(f"Copied {sp} -> {dp}")
            else:
                print(f"Skip {fn} (exists).")
    print("✅ Copy completed (only new files copied).")

def safe_rename(rename_map: Dict[str, str], folder_path: Path) -> None:
    """
    Safely rename files inside a given folder according to a mapping of {old_name: new_name}.
    Uses a two-step temporary renaming process to avoid overwriting.

    Parameters
    ----------
    rename_map : dict
        A dictionary mapping old filenames to new filenames.
    folder_path : str
        Path to the folder where the files are located.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist.")
    temps = {}
    for old, new in rename_map.items():
        op = folder_path / old
        if op.exists():
            tp = op.with_suffix(op.suffix + ".tmp")
            os.rename(op, tp)
            temps[tp] = folder_path / new
            print(f"Temp rename {old} -> {tp.name}")
        else:
            print(f"⚠️ Missing {old}, skip")
    for tp, fp in temps.items():
        if fp.exists():
            print(f"⚠️ Target exists {fp.name}, skip")
            continue
        os.rename(tp, fp); print(f"Renamed {tp.name} -> {fp.name}")
    print("✅ Safe renames done.")

def delete_unlisted_files(directory: Path, keep_files: Iterable[Path]) -> None:
    """
    Delete all files from 'directory' that are NOT in the 'keep_files' list.

    Parameters
    ----------
    directory : str
        Path to the folder where files are located.
    keep_files : list
        Full paths of files to keep (everything else will be deleted).
    """
    directory = Path(directory)
    keep = {Path(os.path.abspath(str(p))) for p in keep_files}
    for fn in os.listdir(directory):
        p = Path(os.path.abspath(str(directory / fn)))
        if p.is_file():
            if p not in keep:
                p.unlink(); print(f"🗑️ Deleted {p}")
            else:
                print(f"✅ Keeping {p}")
    print("✨ Cleanup complete.")

def prepare_htberman(*, do_copy=True, do_rename=True, do_cleanup=True) -> None:
    if do_copy:
        copy_new_files(configs.Dirs.htberman_raw_hebrew,
                       configs.Dirs.htberman_processed_hebrew)

    if do_rename:
        rename_map = {
            "hag108j.txt": "hag109f1.txt",
            "leo201a.txt": "leo201c.txt",
            "leo201b.txt": "leo201a.txt",
            "leo201c.txt": "leo201b.txt",
            "leo202c.txt": "leo202d.txt",
            "leo202d.txt": "leo202e.txt",
            "leo202e.txt": "leo202f.txt",
            "leo202f.txt": "leo202c.txt",
            }
        safe_rename(rename_map, configs.Dirs.htberman_processed_hebrew)

    if do_cleanup:
        d = configs.Dirs.htberman_processed_hebrew
        keep = sorted(
            (d / fn).resolve()
            for fn in os.listdir(d)
            if fn.endswith(".txt")
            and not any(k in fn for k in ["log", "morph", "tmp"])
            and any(k in fn for k in ["hag", "sma", "lio", "leo"])
        )
        delete_unlisted_files(d, keep)

if __name__ == "__main__":
    prepare_htberman()
