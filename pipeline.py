import laspy
import numpy as np
import argparse
import subprocess
from scipy.spatial import cKDTree
from pathlib import Path

FOV = 45
W = 1024
H = 512

def removeOutliers(positions, k=16, stdRatio=3.0):
    tree = cKDTree(positions)
    dist, _ = tree.query(positions, k=k+1, workers=-1)
    meanDist = dist[:, 1:].mean(axis=1)
    return meanDist < meanDist.mean() + stdRatio * meanDist.std()

def readPointCloud(path):
    las = laspy.read(path)
    
    print(las.header.point_format.id, las.header.point_count)
    print(list(las.point_format.dimension_names))

    # Coordinates
    x, y, z = np.asarray(las.x), np.asarray(las.z), -np.asarray(las.y)
    # Colors
    rgb = np.vstack((las.red, las.green, las.blue)).T / 65535.0

    keep = removeOutliers(np.column_stack((x, y, z)))
    x, y, z, rgb = x[keep], y[keep], z[keep], rgb[keep]
    xMax, xMin = x.max(), x.min()
    yMax, yMin = y.max(), y.min()
    zMax, zMin = z.max(), z.min()
    x = x - (xMin + xMax) / 2
    z = z - (zMin + zMax) / 2
    y = y - yMin
    xyz = np.vstack((x, y, z)).T.astype(np.float32)
    bbMin, bbMax = xyz.min(axis=0), xyz.max(axis=0)


    return xyz, rgb, bbMin, bbMax

def generateDataset(path, out, alpha):
    positions, colors, bbMin, bbMax = readPointCloud(path)

    pos = positions
    rand = np.random.default_rng(0)
    if len(pos) > 500_000:
        pos = pos[rand.choice(len(pos), 500_000, replace=False)]

    mean = pos.mean(axis=0)
    cov = np.cov((pos - mean).T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]

    s = np.sqrt(eigvals)
    planarity = (s[1] - s[2]) / s[0]
    sphericity = s[2] / s[0]

    # Determine cameras
    # Camera in the middle 
    a = bbMax[0] - bbMin[0]
    b = bbMax[2] - bbMin[2]
    c = bbMax[1] - bbMin[1]
    center = c / 2
    target = [0, center, 0]
    ha = np.cos(alpha) * (a / (2 * np.sin(alpha)))
    hb = np.cos(alpha) * (b / (2 * np.sin(alpha)))
    hc = np.cos(alpha) * (c / (2 * np.sin(alpha)))

    print(planarity, sphericity)
    h = np.max([ha, hb, hc, a/2, b/2, center]) + center
    if (planarity < sphericity):
        cmd = [
            "deno", "task", "capture", out,
            f"--lasFile={path}",
            f"--width={W}",
            f"--height={H}",
            f"--targetX={target[0]}",
            f"--targetY={target[1]}",
            f"--targetZ={target[2]}",
            f"--mode=DISKS",
            f"--pointSize=0.3",
            f"--noCams=250",
            f"--radius={h}",
            f"--fullSphera=1"
        ]
        subprocess.run(cmd, check=True)
    else:
        cmd = [
            "deno", "task", "capture", out,
            f"--lasFile={path}",
            f"--width={W}",
            f"--height={H}",
            f"--targetX={target[0]}",
            f"--targetY={target[1]}",
            f"--targetZ={target[2]}",
            f"--mode=DISKS",
            f"--pointSize=0.3",
            f"--noCams=150",
            f"--radius={h}",
        ]
        subprocess.run(cmd, check=True)
        for i in [1, -1]:
            for j in [1, -1]:
                cmd = [
                    "deno", "task", "capture", out,
                    f"--lasFile={path}",
                    f"--width={W}",
                    f"--height={H}",
                    f"--targetX={target[0] + (i * a/4)}",
                    f"--targetY={target[1]}",
                    f"--targetZ={target[2] + (j * b/4)}",
                    f"--mode=DISKS",
                    f"--pointSize=0.3",
                    f"--noCams=150",
                    f"--radius={h/2}",
                ]
                subprocess.run(cmd, check=True)
def runGaussianSplattin(out):
    cmd = [
        "conda", "run", "-n", "gaussian_splatting", "--no-capture-output",
        "python", "./gaussian_splatting/train.py", "-s", out,
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lasFile")
    parser.add_argument("--iteration", type=int, default=3)

    args = parser.parse_args()
    halfV = np.deg2rad(FOV / 2)
    halfH = np.arctan(np.tan(halfV) * W / H)

    out = f"data/{Path(args.lasFile).stem}" 
    generateDataset(args.lasFile, out, halfH)


if __name__ == "__main__":
    main()