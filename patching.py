##### patching.py #####
import numpy as np
from typing import List, Tuple

def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """
    Compute patch starting positions with minimal overlap.
    """
    if dimension_size <= patch_size:
        return [0]

    n_patches = np.ceil(dimension_size / patch_size)
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    return [max(0, int(i * (patch_size - total_overlap))) for i in range(int(n_patches))]

def extract_3d_patches(arrays: List[np.ndarray], patch_size: int):
    """
    Extract minimal-overlap 3D patches.
    """
    m, n, l = arrays[0].shape
    x_starts = calculate_patch_starts(m, patch_size)
    y_starts = calculate_patch_starts(n, patch_size)
    z_starts = calculate_patch_starts(l, patch_size)

    patches, coordinates = [], []
    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patches.append(arr[x:x+patch_size, y:y+patch_size, z:z+patch_size])
                    coordinates.append((x, y, z))

    return patches, coordinates

