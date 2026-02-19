"""
Tiling script for landformAI semantic segmentation pipeline.

Tiles input rasters (image + mask) with configurable overlap and splits
into train/valid sets with a spatial buffer to prevent data leakage.

Requirements: rasterio, numpy, scikit-learn
    pip install rasterio numpy scikit-learn

Usage:
    python tile_and_split.py \
        --img /path/to/image.tif \
        --mask /path/to/mask.tif \
        --out /path/to/output \
        --tile_size 256 \
        --overlap 0.25 \
        --val_ratio 0.2 \
        --buffer 1 \
        --seed 42
"""

import argparse
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from sklearn.model_selection import train_test_split


def compute_tile_origins(raster_size, tile_size, overlap):
    """
    Compute top-left pixel coordinates for tiles with given overlap.
    
    Parameters
    ----------
    raster_size : int
        Size of the raster in one dimension (height or width in pixels).
    tile_size : int
        Size of each tile (pixels).
    overlap : float
        Fractional overlap (e.g. 0.25 for 25%).
    
    Returns
    -------
    list of int
        Top-left pixel coordinates along this axis.
    """
    stride = int(tile_size * (1 - overlap))
    origins = list(range(0, raster_size - tile_size + 1, stride))
    # ensure the last tile is included (edge case)
    if len(origins) == 0 or origins[-1] + tile_size < raster_size:
        origins.append(max(0, raster_size - tile_size))
    return origins


def tile_raster(img_path, mask_path, out_dir, tile_size=256, overlap=0.25,
                val_ratio=0.2, buffer_tiles=1.0, seed=42):
    """
    Tile image and mask rasters with overlap, then split into train/valid
    sets with a spatial buffer zone to prevent leakage.

    Strategy:
    ---------
    1. Compute a regular grid of non-overlapping "blocks" (stride = tile_size).
    2. Randomly assign blocks to train or valid set.
    3. For each valid block, compute a pixel exclusion zone of
       (buffer_tiles * tile_size) px around its boundary. Any train tile
       whose footprint intrudes into this zone is excluded (skipped).
    4. Generate overlapping tiles only within assigned (and non-buffered) regions.

    Parameters
    ----------
    img_path : str
        Path to the input image raster (.tif).
    mask_path : str
        Path to the corresponding mask raster (.tif).
    out_dir : str
        Output root directory. Will contain train/ and valid/ subdirs.
    tile_size : int
        Tile size in pixels (square tiles).
    overlap : float
        Fractional overlap between tiles (0.0 - 0.5).
    val_ratio : float
        Approximate fraction of blocks assigned to validation.
    buffer_tiles : float
        Buffer width expressed as a fraction of tile_size.
        0   = no buffer (train tiles may overlap valid pixels).
        0.5 = half-tile buffer (good compromise, ~20% tiles excluded).
        1.0 = full-tile buffer (maximum independence, ~40% tiles excluded).
    seed : int
        Random seed for reproducibility.
    """

    train_dir = os.path.join(out_dir, "train")
    valid_dir = os.path.join(out_dir, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    with rasterio.open(img_path) as img_src, rasterio.open(mask_path) as mask_src:
        
        height = img_src.height
        width = img_src.width

        assert img_src.height == mask_src.height and img_src.width == mask_src.width, \
            f"Image ({img_src.height}x{img_src.width}) and mask ({mask_src.height}x{mask_src.width}) dimensions must match."

        print(f"Raster size: {width} x {height} px")
        print(f"Image bands: {img_src.count}, Mask bands: {mask_src.count}")
        print(f"Tile size: {tile_size}, Overlap: {overlap*100:.0f}%")

        # --- Step 1: Define non-overlapping block grid for spatial splitting ---
        block_rows = list(range(0, height - tile_size + 1, tile_size))
        block_cols = list(range(0, width - tile_size + 1, tile_size))
        
        if not block_rows or block_rows[-1] + tile_size < height:
            block_rows.append(max(0, height - tile_size))
        if not block_cols or block_cols[-1] + tile_size < width:
            block_cols.append(max(0, width - tile_size))

        n_block_rows = len(block_rows)
        n_block_cols = len(block_cols)
        total_blocks = n_block_rows * n_block_cols

        print(f"Block grid: {n_block_rows} rows x {n_block_cols} cols = {total_blocks} blocks")

        # --- Step 2: Assign blocks to train/valid ---
        block_indices = np.arange(total_blocks)
        np.random.seed(seed)
        
        _, val_indices = train_test_split(
            block_indices, test_size=val_ratio, random_state=seed
        )
        val_set = set(val_indices.tolist())

        # --- Step 3: Build pixel-level exclusion zones around valid blocks ---
        def idx_to_rc(idx):
            return idx // n_block_cols, idx % n_block_cols

        def rc_to_idx(r, c):
            return r * n_block_cols + c

        # Pixel buffer distance: fraction of tile_size
        buffer_px = int(buffer_tiles * tile_size)

        # For each valid block, record its pixel extent expanded by buffer_px
        # A train tile is excluded if its footprint overlaps any exclusion zone.
        val_exclusion_zones = []   # list of (r_min, r_max, c_min, c_max) in pixels
        for vi in val_set:
            vr, vc = idx_to_rc(vi)
            r0 = block_rows[vr]
            c0 = block_cols[vc]
            val_exclusion_zones.append((
                r0 - buffer_px,
                r0 + tile_size + buffer_px,
                c0 - buffer_px,
                c0 + tile_size + buffer_px,
            ))

        def tile_in_exclusion_zone(row, col):
            """Return True if the tile at (row, col) overlaps any valid exclusion zone."""
            if buffer_px == 0:
                return False
            t_r0, t_r1 = row, row + tile_size
            t_c0, t_c1 = col, col + tile_size
            for (z_r0, z_r1, z_c0, z_c1) in val_exclusion_zones:
                if t_r0 < z_r1 and t_r1 > z_r0 and t_c0 < z_c1 and t_c1 > z_c0:
                    return True
            return False

        train_set = set(block_indices.tolist()) - val_set

        print(f"Blocks -> train: {len(train_set)}, valid: {len(val_set)}")
        print(f"Buffer: {buffer_tiles} x tile_size = {buffer_px} px exclusion zone around valid blocks")

        # --- Step 4: Generate overlapping tiles within each assigned region ---
        stride = int(tile_size * (1 - overlap))
        
        # Precompute which block each tile origin falls into
        def get_block_idx(row, col):
            """Find which block a given pixel coordinate belongs to."""
            br = min(int(row // tile_size), n_block_rows - 1)
            bc = min(int(col // tile_size), n_block_cols - 1)
            # handle edge blocks
            for i, br_origin in enumerate(block_rows):
                if row < br_origin + tile_size:
                    br = i
                    break
            for j, bc_origin in enumerate(block_cols):
                if col < bc_origin + tile_size:
                    bc = j
                    break
            return rc_to_idx(br, bc)

        # For overlapping tiles: a tile is assigned to valid only if its
        # center falls within a valid block. For training, the tile center
        # must fall within a train block (not buffer, not valid).
        tile_origins_row = compute_tile_origins(height, tile_size, overlap)
        tile_origins_col = compute_tile_origins(width, tile_size, overlap)

        n_train = 0
        n_valid = 0
        n_skipped = 0
        basename = Path(img_path).stem

        for row in tile_origins_row:
            for col in tile_origins_col:
                # Determine assignment based on tile center
                center_r = row + tile_size // 2
                center_c = col + tile_size // 2

                # Find block index for tile center
                br = 0
                for i, br_origin in enumerate(block_rows):
                    if center_r < br_origin + tile_size:
                        br = i
                        break
                bc = 0
                for j, bc_origin in enumerate(block_cols):
                    if center_c < bc_origin + tile_size:
                        bc = j
                        break
                block_idx = rc_to_idx(br, bc)

                if block_idx in val_set:
                    target_dir = valid_dir
                    n_valid += 1
                elif block_idx in train_set:
                    if tile_in_exclusion_zone(row, col):
                        # tile footprint overlaps a valid exclusion zone — skip
                        n_skipped += 1
                        continue
                    target_dir = train_dir
                    n_train += 1
                else:
                    n_skipped += 1
                    continue

                # Read tile data
                window = Window(col_off=col, row_off=row, width=tile_size, height=tile_size)
                
                img_tile = img_src.read(window=window)   # (bands, H, W)
                mask_tile = mask_src.read(window=window)  # (bands, H, W)

                # Skip tiles that are entirely nodata / zero
                if img_tile.max() == 0 and img_tile.min() == 0:
                    n_skipped += 1
                    continue

                tile_name = f"{basename}_r{row}_c{col}"
                img_out = os.path.join(target_dir, f"{tile_name}.tif")
                mask_out = os.path.join(target_dir, f"{tile_name}_mask.tif")

                # Write image tile
                profile = img_src.profile.copy()
                profile.update(
                    width=tile_size,
                    height=tile_size,
                    transform=rasterio.windows.transform(window, img_src.transform)
                )
                with rasterio.open(img_out, 'w', **profile) as dst:
                    dst.write(img_tile)

                # Write mask tile
                mask_profile = mask_src.profile.copy()
                mask_profile.update(
                    width=tile_size,
                    height=tile_size,
                    transform=rasterio.windows.transform(window, mask_src.transform)
                )
                with rasterio.open(mask_out, 'w', **mask_profile) as dst:
                    dst.write(mask_tile)

        print(f"\nDone. Tiles written:")
        print(f"  Train: {n_train}")
        print(f"  Valid: {n_valid}")
        print(f"  Skipped (buffer/nodata): {n_skipped}")
        print(f"\nOutput directories:")
        print(f"  {train_dir}")
        print(f"  {valid_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tile rasters for semantic segmentation with spatial train/val split."
    )
    parser.add_argument("--img", required=True, help="Path to input image raster (.tif)")
    parser.add_argument("--mask", required=True, help="Path to mask/label raster (.tif)")
    parser.add_argument("--out", required=True, help="Output directory (will contain train/ and valid/)")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size in pixels (default: 256)")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap fraction (default: 0.25)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    parser.add_argument("--buffer", type=float, default=1.0,
                        help="Buffer as fraction of tile_size around valid blocks (0=none, 0.5=half-tile, 1.0=full-tile, default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    tile_raster(
        img_path=args.img,
        mask_path=args.mask,
        out_dir=args.out,
        tile_size=args.tile_size,
        overlap=args.overlap,
        val_ratio=args.val_ratio,
        buffer_tiles=args.buffer,
        seed=args.seed
    )
