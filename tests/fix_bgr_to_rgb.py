import numpy as np
import sys
from pathlib import Path


def convert_bgr_to_rgb(src: str, dst: str) -> None:
    data = np.loadtxt(src)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 3:
        raise ValueError(
            f"Expected 3 columns (BGR), got {data.shape[1]}. "
            f"This script only converts 3-band samples."
        )
    rgb = data[:, [2, 1, 0]]
    fmt = "%.18e" if data.dtype == float else "%d"
    np.savetxt(dst, rgb, fmt=fmt)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src_bgr.txt> <dst_rgb.txt>", file=sys.stderr)
        sys.exit(1)
    convert_bgr_to_rgb(sys.argv[1], sys.argv[2])
    print(f"Converted {sys.argv[1]} -> {sys.argv[2]}")
