import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tqdm import tqdm
from omegaconf import OmegaConf
import yaml
import importlib
from termcolor import colored

# -----------------------------
# Paths & imports
# -----------------------------
sys.path.append(".")
sys.path.append("/content/ce-vae-underwater-image-enhancement/Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2

torch.backends.cudnn.benchmark = True
PRINT_COLOR = "red"

# ============================================================
# Depth Anything v2 (GLOBAL, GPU, MAIN PROCESS ONLY)
# ============================================================

DEPTH_DEVICE = "cuda"

depth_model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
).to(DEPTH_DEVICE)

depth_model.load_state_dict(
    torch.load(
        "/content/ce-vae-underwater-image-enhancement/depth_anything_v2_vits.pth", # CHANGE
        map_location=DEPTH_DEVICE,
    )
)
depth_model.eval()


@torch.inference_mode()
def infer_depth_batch(pil_images, target_size=256):
    """
    pil_images: list[PIL.Image]
    returns: torch.Tensor (B, 1, H, W) in [-1, 1]
    """
    depths = []

    for img in pil_images:
        img_np = np.array(img)

        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        depth = depth_model.infer_image(img_np).astype(np.float32)

        # normalize per-image
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        # map to JOINT-ID range
        depth = depth * 288.0

        # resize
        depth = Image.fromarray(depth).resize(
            (target_size, target_size), Image.NEAREST
        )
        depth = np.array(depth, dtype=np.float32)

        # normalize to [-1, 1]
        depth = depth / 144.0 - 1.0

        depths.append(torch.from_numpy(depth).unsqueeze(0))

    return torch.stack(depths, dim=0).to(DEPTH_DEVICE)


# ============================================================
# Dataset (NO DEPTH, NO PIL RETURN)
# ============================================================

def is_image(p: Path):
    return p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]


def preprocess_rgb(img: Image.Image, size=256):
    img = TF.resize(img, (size, size), interpolation=Image.LANCZOS)
    img = T.ToTensor()(img)
    return img * 2.0 - 1.0


class ImageFolderDataset(Dataset):
    def __init__(self, root, size=256):
        self.size = size
        self.paths = sorted([p for p in Path(root).glob("*") if is_image(p)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        rgb = preprocess_rgb(img, self.size)
        return rgb, str(path)


# ============================================================
# Model loading (CONFIG IS READ ONLY)
# ============================================================

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def load_config(cfg_path):
    return OmegaConf.load(cfg_path)


def load_cevae(config, ckpt_path):
    cls = get_obj_from_str(config.model.target)

    # IMPORTANT: DO NOT MODIFY CONFIG FILE
    params = dict(config.model.params)
    params.pop("ckpt_path", None)
    params.pop("lossconfig", None)
    params.pop("optimizer", None)

    model = cls(**params)

    print(colored(f"Loading checkpoint {ckpt_path}", PRINT_COLOR))
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)

    return model.eval()


# ============================================================
# Utils
# ============================================================

def tensor_to_pil(x):
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((x * 255).astype(np.uint8))


# ============================================================
# Main inference loop
# ============================================================

@torch.inference_mode()
def run(opt):
    device = torch.device(opt.device)

    config = load_config(opt.config)
    model = load_cevae(config, opt.checkpoint).to(device)

    print(colored(f"Running inference on {opt.data_path}", PRINT_COLOR))

    out_dir = Path(opt.output_path)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    dataset = ImageFolderDataset(opt.data_path)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=2,   # SAFE
        pin_memory=True
    )

    print(colored(f"Processing {len(dataset)} images", PRINT_COLOR))

    for rgb, paths in tqdm(loader):
        rgb = rgb.to(device)

        # reopen PIL in main process
        pil_imgs = [Image.open(p).convert("RGB") for p in paths]

        depth = infer_depth_batch(pil_imgs)

        x = torch.cat([rgb, depth], dim=1)

        out = model(x)

        for img, p in zip(out, paths):
            tensor_to_pil(img).save(out_dir / Path(p).name)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", default="./output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")

    opt = parser.parse_args()

    print(colored("Running with args:", PRINT_COLOR))
    print(colored(opt, PRINT_COLOR))

    run(opt)
