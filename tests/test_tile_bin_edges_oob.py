"""
Test that get_tile_bin_edges correctly handles the case where
num_tiles > num_intersects.
"""

import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_tile_bins_oob_when_num_tiles_exceeds_num_intersects():
    from gsplat.utils import get_tile_bin_edges

    # Suppose we have a 512x512 image with 16x16 tiles => 32x32 = 1024 tiles.
    # But only 3 gaussians, each hitting exactly 1 tile, so num_intersects = 3.
    # the kernel writes to tile_bins[tile_idx] where tile_idx can be up to 1023

    num_tiles = 1024  # 32 x 32 tile grid
    num_intersects = 3

    # Craft isect_ids_sorted: each entry is (tile_id << 32 | depth_id).
    # Place intersections in tiles 0, 500, and 1023 (the last tile).
    tile_ids = torch.tensor([0, 500, 1023], dtype=torch.int64, device=device)
    depth_ids = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
    isect_ids_sorted = (tile_ids << 32) | depth_ids

    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, num_tiles)

    # Verify the output is large enough to index by any tile.
    assert tile_bins.shape[0] >= num_tiles, (
        f"tile_bins has {tile_bins.shape[0]} rows but we need at least {num_tiles} "
        f"to safely index by tile_id"
    )

    # Verify correctness of the bin edges for the tiles that have intersections.
    # tile 0: intersects [0, 1)
    assert tile_bins[0, 0].item() == 0
    assert tile_bins[0, 1].item() == 1

    # tile 500: intersects [1, 2)
    assert tile_bins[500, 0].item() == 1
    assert tile_bins[500, 1].item() == 2

    # tile 1023: intersects [2, 3)
    assert tile_bins[1023, 0].item() == 2
    assert tile_bins[1023, 1].item() == 3

    # Tiles with no intersections should have [0, 0] (empty range).
    assert tile_bins[1, 0].item() == 0 and tile_bins[1, 1].item() == 0
    assert tile_bins[999, 0].item() == 0 and tile_bins[999, 1].item() == 0
