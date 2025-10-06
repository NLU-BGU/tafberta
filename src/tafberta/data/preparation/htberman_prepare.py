from typing import List
import pylangacq
from pylangacq import Reader
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
import string
import os
import shutil

# from tafberta.utils import get_children, get_not_child_participants, get_child_participants, is_dataset_has_morphology
from tafberta import configs


def copy_new_files(src_dir: str, dst_dir: str) -> None:
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
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
    
    os.makedirs(dst_dir, exist_ok=True)

    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        if os.path.isfile(src_path):
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} -> {dst_path}")
            else:
                print(f"Skipping {filename}, already exists in destination.")
        else:
            print(f"Skipping {src_path} (not a file).")

    print("✅ Copy completed (only new files copied).")



def safe_rename(rename_map: dict, folder_path: str) -> None:
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
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist.")

    temp_map = {}

    # Step 1: rename originals to temporary files
    for old, new in rename_map.items():
        old_path = os.path.join(folder_path, old)
        if os.path.exists(old_path):
            tmp_path = old_path + ".tmp"
            os.rename(old_path, tmp_path)
            temp_map[tmp_path] = os.path.join(folder_path, new)
            print(f"Temporarily renamed {old} -> {old}.tmp")
        else:
            print(f"⚠️ Skipping {old}, file not found in {folder_path}")

    # Step 2: rename temp files to final names
    for tmp_path, final_path in temp_map.items():
        if os.path.exists(final_path):
            print(f"⚠️ Skipping {tmp_path}, target {os.path.basename(final_path)} already exists.")
            continue
        os.rename(tmp_path, final_path)
        print(f"Renamed {os.path.basename(tmp_path)} -> {os.path.basename(final_path)}")

    print("✅ All renames completed safely.")





def delete_unlisted_files(directory: str, keep_files: list) -> None:
    """
    Delete all files from 'directory' that are NOT in the 'keep_files' list.

    Parameters
    ----------
    directory : str
        Path to the folder where files are located.
    keep_files : list
        Full paths of files to keep (everything else will be deleted).
    """
    keep_set = set(os.path.abspath(f) for f in keep_files)

    for filename in os.listdir(directory):
        file_path = os.path.abspath(os.path.join(directory, filename))

        if os.path.isfile(file_path):
            if file_path not in keep_set:
                os.remove(file_path)
                print(f"🗑️ Deleted {file_path}")
            else:
                print(f"✅ Keeping {file_path}")
        else:
            print(f"Skipping {file_path} (not a file).")

    print("✨ Cleanup complete.")




# Copy all files from raw htberman to processed htberman
# copy_new_files(configs.Dirs.htberman_raw_hebrew, configs.Dirs.htberman_processed_hebrew)

# Fix known filename conflicts in HTBerman dataset
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
# safe_rename(rename_map, configs.Dirs.htberman_processed_hebrew)


# Delete all files from htberman_processed_hebrew that are NOT in the list of speech files
directory = configs.Dirs.htberman_processed_hebrew

speech_files = sorted(
    os.path.join(directory, filename)
    for filename in os.listdir(directory)
    if not any(f_type in os.path.join(directory, filename) for f_type in ['log', 'morph', 'tmp'])
    and any(f_type in os.path.join(directory, filename) for f_type in ['txt'])
    and any(f_type in os.path.join(directory, filename) for f_type in ['hag', 'sma', 'lio', 'leo'])
)

# delete_unlisted_files(directory, speech_files)



# TODO: write as a function
# Load original Hebrew CHILDES datasets (English transcription)
# if os.path.exists(os.path.join(configs.Dirs.htberman_raw_english, "datasets.p")):
#     with open(os.path.join(configs.Dirs.htberman_raw_english, "datasets.p"), "rb") as f:
#         datasets = pickle.load(f)
# else:
#     datasets = {
#         ds: pylangacq.read_chat(configs.DataPrep.url % ds)
#         for ds in configs.DataPrep.all_datasets
#     }
#     with open(os.path.join(configs.Dirs.htberman_raw_english, "datasets.p"), "wb") as f:
#         pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)


# Load BermanLong dataset (English transcription)
reader = pylangacq.read_chat(configs.DataPrep.url % configs.DataPrep.childes_in_htberman_dataset)
readers = dict(zip(reader.file_paths(),
                   [reader.pop_left() for i in range(len(reader.file_paths()))]))

empty_file = 'BermanLong/Leor/leo300e.cha'
del readers[empty_file]

with open(os.path.join(configs.Dirs.htberman_raw_english, "readers_BermanLong.p"), "wb") as f:
    pickle.dump(readers, f, protocol=pickle.HIGHEST_PROTOCOL)
