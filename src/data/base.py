from typing import Tuple, Any, Dict, List
import cv2
import numpy as np
import albumentations as A
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_rgba(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "RGBA":
        raise ValueError(f"Expected RGBA image (RGB+Depth), got {img.mode}")
    return np.array(img, dtype=np.uint8)


class ImagePairDatasetFromPaths(Dataset):
    def __init__(
        self,
        paths: List[str],
        path_target: List[str],
        size: int = None,
        random_crop: bool = False,
        random_flip: bool = False,
        max_size: int = None,
        color_jitter: dict = None,
        perspective: dict = None,
        scale: dict = None,
        is_test: bool = False,
    ):
        self.paths = paths
        self.path_target = path_target
        self._length = len(paths)
        self.is_test = is_test

        # -------------------------------
        # GEOMETRIC TRANSFORMS (RGB+Depth+Target)
        # -------------------------------
        geom = []

        if size is not None:
            geom.append(
                A.SmallestMaxSize(
                    max_size=size if max_size is None else max_size
                )
            )

            if scale is not None:
                geom.append(A.RandomScale(p=1.0, **scale))

            if random_crop:
                geom.append(A.RandomCrop(size, size))
            else:
                geom.append(A.CenterCrop(size, size))

            if random_flip:
                geom.append(A.HorizontalFlip(p=0.5))

            if perspective is not None:
                geom.append(A.Perspective(p=0.5, **perspective))

        self.geom_tf = A.Compose(
            geom,
            additional_targets={
                "depth": "image",
                "target": "image",
            },
            is_check_shapes=False,
        )

        # -------------------------------
        # COLOR TRANSFORMS (RGB ONLY)
        # -------------------------------
        if not is_test and color_jitter is not None:
            self.color_tf = A.ColorJitter(p=0.5, **color_jitter)
        else:
            self.color_tf = None

    def __len__(self):
        return self._length

    def preprocess_images(
        self, image_path: str, target_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        rgba = _load_rgba(image_path)
        rgb = rgba[..., :3]
        depth = rgba[..., 3:4]

        target = _load_rgb(target_path)

        # ---- GEOMETRIC (shared) ----
        transformed = self.geom_tf(
            image=rgb,
            depth=depth,
            target=target,
        )

        rgb = transformed["image"]
        depth = transformed["depth"]
        target = transformed["target"]

        # ---- COLOR (RGB ONLY) ----
        if self.color_tf is not None:
            rgb = self.color_tf(image=rgb)["image"]

        # ---- NORMALIZATION ----
        rgb = rgb.astype(np.float32) / 127.5 - 1.0
        target = target.astype(np.float32) / 127.5 - 1.0

        depth = depth.astype(np.float32)
        depth = depth / (depth.max() + 1e-6)

        # ---- CONCAT RGB + DEPTH ----
        image = np.concatenate([rgb, depth], axis=-1)

        return image, target

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, target = self.preprocess_images(
            self.paths[idx],
            self.path_target[idx],
        )

        return {
            "image": image,
            "target": target,
            "file_path_": self.paths[idx],
            "file_path_target": self.path_target[idx],
        }
