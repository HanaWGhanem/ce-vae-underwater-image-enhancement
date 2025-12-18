import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import lpips
from skimage.metrics import structural_similarity as compare_ssim
import math
import csv

# ------------------------------
# Helper functions
# ------------------------------
def psnr(img1, img2):
    """Compute PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim_safe(img1, img2):
    """Compute SSIM safely for any image size"""
    h, w = img1.shape[:2]
    win_size = min(7, h, w)  # must be odd and ≤ smaller image side
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 1:
        win_size = 1
    return compare_ssim(img1, img2, data_range=255, channel_axis=-1, win_size=win_size)

def load_image(path):
    """Load image as uint8 numpy array"""
    img = Image.open(path).convert("RGB")
    return np.array(img)

# ------------------------------
# Metrics calculation
# ------------------------------
def compute_metrics(ref_path, pred_path, lpips_model, device):
    ref = load_image(ref_path)
    pred = load_image(pred_path)

    # Resize predicted image to match reference size
    if pred.shape != ref.shape:
        pred = np.array(Image.fromarray(pred).resize((ref.shape[1], ref.shape[0]), Image.LANCZOS))

    # PSNR
    psnr_val = psnr(ref, pred)

    # SSIM
    ssim_val = ssim_safe(ref, pred)

    # LPIPS
    # Convert to tensor in [-1,1], shape [1,3,H,W]
    ref_t = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
    pred_t = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
    with torch.no_grad():
        lpips_val = lpips_model(ref_t, pred_t).item()

    return psnr_val, ssim_val, lpips_val

# ------------------------------
# Main CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute PSNR, SSIM, LPIPS metrics")
    parser.add_argument("--ref", type=str, required=True, help="Reference image or folder")
    parser.add_argument("--pred", type=str, required=True, help="Predicted image or folder")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for LPIPS computation")
    parser.add_argument("--csv", type=str, default="metrics.csv", help="Output CSV file for per-image metrics")
    args = parser.parse_args()

    ref_path = Path(args.ref)
    pred_path = Path(args.pred)

    # Load LPIPS
    lpips_model = lpips.LPIPS(net='vgg').to(args.device)
    lpips_model.eval()

    # Gather file lists
    if ref_path.is_dir():
        ref_files = sorted([f for f in ref_path.glob("*") if f.suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tiff",".tif"]])
    else:
        ref_files = [ref_path]

    if pred_path.is_dir():
        pred_files = sorted([f for f in pred_path.glob("*") if f.suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tiff",".tif"]])
    else:
        pred_files = [pred_path]

    # If single ref image but multiple preds, repeat ref
    if len(ref_files) == 1 and len(pred_files) > 1:
        ref_files = ref_files * len(pred_files)

    assert len(ref_files) == len(pred_files), "Number of reference and predicted images must match!"

    # Accumulate metrics
    psnr_list, ssim_list, lpips_list = [], [], []

    # Open CSV
    with open(args.csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_name", "PSNR", "SSIM", "LPIPS"])

        print(f"Processing {len(ref_files)} images...")
        for r, p in tqdm(zip(ref_files, pred_files), total=len(ref_files)):
            ps, ss, lp = compute_metrics(r, p, lpips_model, args.device)
            psnr_list.append(ps)
            ssim_list.append(ss)
            lpips_list.append(lp)

            csvwriter.writerow([p.name, ps, ss, lp])

    # Print averages
    print("------ Metrics ------")
    print(f"PSNR:  {np.mean(psnr_list):.4f}")
    print(f"SSIM:  {np.mean(ssim_list):.4f}")
    print(f"LPIPS: {np.mean(lpips_list):.4f}")
    print(f"Per-image metrics saved to: {args.csv}")

if __name__ == "__main__":
    main()
