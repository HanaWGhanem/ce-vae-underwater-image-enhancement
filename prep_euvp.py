
import shutil
from pathlib import Path

# -------- Paths --------
SRC_ROOT = Path("euvp-data/EUVP/Paired/underwater_imagenet")
DST_ROOT = Path("output_dataset")

# -------- Mapping --------
splits = {
    "train": {
        "input": SRC_ROOT / "trainA",
        "GT": SRC_ROOT / "trainB",
    },
    "val": {
        "input": SRC_ROOT / "validation" / "trainA",
        "GT": SRC_ROOT / "validation" / "trainB",
    }
}

# -------- Copy --------
for split, folders in splits.items():
    for target_name, src_path in folders.items():
        dst_path = DST_ROOT / split / target_name
        dst_path.mkdir(parents=True, exist_ok=True)

        if not src_path.exists():
            print(f"[WARNING] Missing: {src_path}")
            continue

        for file in src_path.iterdir():
            if file.is_file():
                shutil.copy2(file, dst_path / file.name)

        print(f"[OK] {src_path} → {dst_path}")

print("\nDataset restructuring complete.")
