import os
import cv2
import random
from tqdm import tqdm

BASE_DATASET_PATH = "/path/to/your/dataset_root"   # data
TXT_FILE          = "/path/to/train.txt"               # train
OUTPUT_ROOT       = "/path/to/dataset_out"         # new data dir

TRAIN_RATIO = 0.7
SEED = 42

random.seed(SEED)

# ---------- Create directory structure ----------
def make_dirs():
    for split in ["train", "val"]:
        for sub in ["input", "GT"]:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, sub), exist_ok=True)

# ---------- Read txt file ----------
def read_txt(txt_path):
    samples = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            distorted, clean, depth = parts[:3]

            samples.append({
                "distorted": os.path.join(BASE_DATASET_PATH, distorted),
                "clean":     os.path.join(BASE_DATASET_PATH, clean),
                "depth":     os.path.join(BASE_DATASET_PATH, depth),
                "name": os.path.splitext(os.path.basename(clean))[0]
            })
    return samples

# ---------- Create RGBA input ----------
def create_rgba(distorted_path, depth_path):
    rgb = cv2.imread(distorted_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Failed to load RGB image: {distorted_path}")

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to load depth image: {depth_path}")

    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    # Resize depth if needed
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    # Normalize depth to 0–255 if needed
    if depth.dtype != rgb.dtype:
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth = depth.astype(rgb.dtype)

    rgba = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = depth

    return rgba

# ---------- Process split ----------
def process_split(samples, split_name):
    input_dir = os.path.join(OUTPUT_ROOT, split_name, "input")
    gt_dir    = os.path.join(OUTPUT_ROOT, split_name, "GT")

    for s in tqdm(samples, desc=f"Processing {split_name}"):
        # ----- GT -----
        gt_img = cv2.imread(s["clean"], cv2.IMREAD_COLOR)
        if gt_img is None:
            raise RuntimeError(f"Failed to load GT image: {s['clean']}")

        gt_path = os.path.join(gt_dir, s["name"] + ".png")
        cv2.imwrite(gt_path, gt_img)

        # ----- INPUT (RGBA) -----
        rgba = create_rgba(s["distorted"], s["depth"])
        input_path = os.path.join(input_dir, s["name"] + ".png")
        cv2.imwrite(input_path, rgba)

if __name__ == "__main__":
    make_dirs()

    samples = read_txt(TXT_FILE)
    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(TRAIN_RATIO * n_total)

    train_samples = samples[:n_train]
    val_samples   = samples[n_train:]

    print("===================================")
    print(f"Total samples : {n_total}")
    print(f"Train samples : {len(train_samples)}")
    print(f"Val samples   : {len(val_samples)}")
    print("===================================")

    process_split(train_samples, "train")
    process_split(val_samples, "val")

    print("\n✅ Dataset preparation complete")
