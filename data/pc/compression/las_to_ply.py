# Convert a single LAS/LAZ file to a binary PLY suitable for Draco / G-PCC.

import sys
import json
import numpy as np
import laspy

POSITION_SCALE = 1000.0  # milimeter


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    las = laspy.read(in_path)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    n = len(x)

    origin = np.array([x.min(), y.min(), z.min()])
    x_mm = np.round((x - origin[0]) * POSITION_SCALE).astype("<f4")
    y_mm = np.round((y - origin[1]) * POSITION_SCALE).astype("<f4")
    z_mm = np.round((z - origin[2]) * POSITION_SCALE).astype("<f4")

    max_extent_mm = max(x_mm.max(), y_mm.max(), z_mm.max())

    has_color = all(hasattr(las, dim) for dim in ("red", "green", "blue"))

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float32 x",
        "property float32 y",
        "property float32 z",
    ]
    if has_color:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines.append("end_header")

    with open(out_path, "wb") as f:
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))

        if has_color:
            r = (np.asarray(las.red, dtype=np.uint32) >> 8).astype("u1")
            g = (np.asarray(las.green, dtype=np.uint32) >> 8).astype("u1")
            b = (np.asarray(las.blue, dtype=np.uint32) >> 8).astype("u1")

            rec = np.zeros(
                n,
                dtype=[
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                ],
            )
            rec["red"], rec["green"], rec["blue"] = r, g, b
        else:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

        rec["x"], rec["y"], rec["z"] = x_mm, y_mm, z_mm
        rec.tofile(f)

    meta = {
        "origin": [float(o) for o in origin],
        "position_scale": POSITION_SCALE,
        "has_color": has_color,
    }
    meta_path = out_path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Wrote {n} points to {out_path} (+ {meta_path}), color: {has_color}, "
        f"origin: {list(origin)}, extent_mm: {max_extent_mm:.0f}"
    )


if __name__ == "__main__":
    main()