"""
inference.py – Full-raster inference for a trained GMM checkpoint.

Pipeline
--------
1. Tile the input raster on-the-fly with 25 % overlap (matching training).
2. Run batched inference through the trained model.
3. Resolve overlapping predictions via majority voting (most robust for
   discrete class labels).
4. Write a single georeferenced GeoTIFF with class labels 1–20, matching
   the original label convention.

Usage
-----
    python inference.py inference_config.yaml

Config keys
-----------
    checkpoint : str   – path to epoch_N folder (model.safetensors + config.json)
    data:
        img  : str     – path to the input feature raster (.tif)
        out  : str     – path for the output prediction raster (.tif)
    model:
        encoder : str  – encoder name (must match checkpoint, e.g. resnet18)
        weights : str  – pretrained weights used in training (e.g. imagenet) or null
        classes : int  – number of output classes (e.g. 20)
    tile_size  : int   – tile size in pixels (default: 256)
    overlap    : float – fractional overlap (default: 0.25)
    batch_size : int   – inference batch size (default: 8)
"""

import argparse
import os
import math
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tile origin helper  (mirrors tile_and_split.py logic)
# ---------------------------------------------------------------------------

def compute_tile_origins(raster_size: int, tile_size: int, overlap: float):
    """Return top-left pixel offsets for tiles with the given overlap."""
    stride = int(tile_size * (1 - overlap))
    origins = list(range(0, raster_size - tile_size + 1, stride))
    if not origins or origins[-1] + tile_size < raster_size:
        origins.append(max(0, raster_size - tile_size))
    return origins


# ---------------------------------------------------------------------------
# Dataset – reads tiles directly from the open rasterio source
# ---------------------------------------------------------------------------

class RasterTileDataset(Dataset):
    """
    Yields (tile_tensor, row_offset, col_offset) for every tile position.
    All reading is done from a single open rasterio dataset handle that is
    shared across workers (num_workers=0 only) to avoid pickling issues.
    """

    def __init__(self, src, tile_size: int, overlap: float, transform):
        self.src       = src
        self.tile_size = tile_size
        self.transform = transform

        row_origins = compute_tile_origins(src.height, tile_size, overlap)
        col_origins = compute_tile_origins(src.width,  tile_size, overlap)

        # All (row, col) tile positions
        self.positions = [(r, c) for r in row_origins for c in col_origins]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        row, col = self.positions[idx]
        window   = Window(col_off=col, row_off=row,
                          width=self.tile_size, height=self.tile_size)

        tile = self.src.read(window=window)          # (C, H, W) uint8 / float32
        tile = np.where(tile < -9999, 0.0, tile).astype(np.float32)  # replace nodata with 0, matching loader.py
        tile = np.transpose(tile, (1, 2, 0))         # HWC for torchvision transform
        tile = np.ascontiguousarray(tile)

        tensor = self.transform(tile)                # (C, H, W) float32
        return tensor, row, col


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str, device: torch.device):
    model = smp.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

def run_inference(cfg: dict):

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_size  = cfg.get("tile_size",  256)
    overlap    = cfg.get("overlap",    0.25)
    batch_size = cfg.get("batch_size", 8)
    n_classes  = cfg["model"]["classes"]

    print(f"Device     : {device}")
    print(f"Tile size  : {tile_size} px  |  Overlap: {overlap*100:.0f}%")
    print(f"Classes    : {n_classes}")

    # -- preprocessing transform (must match training) -----------------------
    encoder   = cfg["model"]["encoder"]
    weights   = cfg["model"].get("weights", "imagenet")
    normalize = cfg["model"].get("normalize", True)  # default True to match original behaviour

    if weights is not None and normalize:
        pp = smp.encoders.get_preprocessing_params(encoder, weights)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pp["mean"], std=pp["std"]),
        ])
    else:
        transform = transforms.ToTensor()  # data already in [0,1]

    # -- model ---------------------------------------------------------------
    checkpoint = cfg["checkpoint"]
    print(f"Loading checkpoint : {checkpoint}")
    model = load_model(checkpoint, device)

    # -- open source raster --------------------------------------------------
    img_path = cfg["data"]["img"]
    print(f"Input raster       : {img_path}")

    with rasterio.open(img_path) as src:

        H, W = src.height, src.width
        print(f"Raster size        : {W} x {H} px")

        # Accumulators for majority voting:
        #   vote_counts[class, row, col]  – how many times each class was predicted
        vote_counts = np.zeros((n_classes, H, W), dtype=np.uint16)

        dataset = RasterTileDataset(src, tile_size, overlap, transform)
        print(f"Total tiles        : {len(dataset)}")

        # num_workers=0 so the open rasterio handle can be shared safely
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

        # -- inference loop --------------------------------------------------
        with torch.no_grad():
            for tiles, rows, cols in tqdm(loader, desc="Inference"):
                tiles = tiles.to(device)                      # (B, C, H, W)

                logits = model(tiles)                         # (B, n_classes, H, W)
                probs  = logits.log_softmax(dim=1).exp()
                preds  = torch.argmax(probs, dim=1)           # (B, H, W)  0-indexed
                preds  = preds.cpu().numpy().astype(np.uint8) # (B, H, W)

                rows = rows.numpy()
                cols = cols.numpy()

                for b in range(preds.shape[0]):
                    r, c = int(rows[b]), int(cols[b])

                    # actual tile extent (may be smaller at edges)
                    th = min(tile_size, H - r)
                    tw = min(tile_size, W - c)

                    pred_tile = preds[b, :th, :tw]            # (th, tw)

                    # scatter votes: one vote per pixel per predicted class
                    np.add.at(
                        vote_counts,
                        (pred_tile,
                         np.arange(th)[:, None] + r,
                         np.arange(tw)[None, :] + c),
                        1,
                    )

    # -- majority vote: argmax over class axis --------------------------------
    print("Resolving overlap via majority voting …")
    prediction = np.argmax(vote_counts, axis=0).astype(np.uint8)  # 0-indexed, (H, W)
    prediction += 1                                                # back to 1-indexed (1…N)

    # -- write output GeoTIFF ------------------------------------------------
    out_path = cfg["data"]["out"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with rasterio.open(img_path) as src:
        profile = src.profile.copy()

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="lzw",
        nodata=None,   # predictions are class labels; no meaningful nodata
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prediction[np.newaxis, :, :])   # add band dim → (1, H, W)

    print(f"Prediction saved   : {out_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM full-raster inference with majority-vote overlap resolution")
    parser.add_argument("config", help="Path to inference YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)
