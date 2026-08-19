import sys
import json
import numpy as np
import laspy
from plyfile import PlyData


def main():
    in_path, meta_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    ply = PlyData.read(in_path)
    vertex = ply["vertex"]
    names = vertex.data.dtype.names

    x_mm = np.asarray(vertex["x"], dtype=np.float64)
    y_mm = np.asarray(vertex["y"], dtype=np.float64)
    z_mm = np.asarray(vertex["z"], dtype=np.float64)

    has_color = all(name in names for name in ("red", "green", "blue"))

    with open(meta_path) as f:
        meta = json.load(f)
    origin = meta["origin"]
    scale = meta["position_scale"]

    x = x_mm / scale + origin[0]
    y = y_mm / scale + origin[1]
    z = z_mm / scale + origin[2]

    header = laspy.LasHeader(point_format=(3 if has_color else 0), version="1.4")
    header.scales = [1.0 / scale] * 3
    header.offsets = origin

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z

    if has_color:
        r8 = np.asarray(vertex["red"], dtype=np.uint32)
        g8 = np.asarray(vertex["green"], dtype=np.uint32)
        b8 = np.asarray(vertex["blue"], dtype=np.uint32)
        # raje pomnožimo z 257 kot 256, da se 255 preslika točno v 65535, in ne 65280.
        las.red = (r8 * 257).astype(np.uint16)
        las.green = (g8 * 257).astype(np.uint16)
        las.blue = (b8 * 257).astype(np.uint16)

    las.write(out_path)
    print(f"Wrote {len(x)} points to {out_path}, color: {has_color}")


if __name__ == "__main__":
    main()