import os
import shutil
import random

def split_and_copy(all_dirs, dest_dir, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15):
    files = []
    for d in all_dirs:
        files.extend([os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])

    random.shuffle(files)

    total = len(files)
    train_end = int(total * train_ratio)
    eval_end = train_end + int(total * eval_ratio)

    splits = {
        "train": files[:train_end],
        "eval": files[train_end:eval_end],
        "test": files[eval_end:]
    }

    for split_name, split_files in splits.items():
        split_dir = os.path.join(dest_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for src in split_files:
            dst = os.path.join(split_dir, os.path.basename(src))
            shutil.copy2(src, dst)

    print(f"Done splitting {len(files)} files into train/eval/test")

if __name__ == "__main__":
    base_dir = r"Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final\data"
    output_dir = r"Final_v1\AIFORTHAI-LST20Corpus\LST20_Corpus_final"

    # Source dirs: all/new, all/old
    source_dirs = [os.path.join(base_dir, "new_fix"), os.path.join(base_dir, "old_fix")]
    split_and_copy(source_dirs, output_dir)
