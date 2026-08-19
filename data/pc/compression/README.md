## 1. Environment setup (Conda + Python)

```bash
# Python environment (conda used here, any venv works)
conda create -n pc_compression python=3.11
conda activate pc_compression
pip install numpy laspy plyfile
```
## 2. Build Draco

```bash
git clone https://github.com/google/draco.git
cd draco
mkdir build && cd build
cmake ..
make
```
## 3. Build G-PCC (TMC13)

```bash
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13
mkdir build && cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make
```
The `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` flag is required for CMake 4.x — TMC13's `CMakeLists.txt` declares an older minimum version that recent CMake refuses to configure without this compatibility override.

Produces `tmc3` in `mpeg-pcc-tmc13/build/tmc3/tmc3`.

## 4. Environment setup (for Mac)

Point at the built binaries (add to `~/.zshrc` to persist across sessions):

```bash
export DRACO_ENCODER="/path/to/draco/build/draco_encoder"
export DRACO_DECODER="/path/to/draco/build/draco_decoder"
export TMC3="/path/to/mpeg-pcc-tmc13/build/tmc3/tmc3"
export PATH="/path/to/draco/build:$PATH"
```

After editing `~/.zshrc`, run `source ~/.zshrc` (or open a new terminal tab)
to pick up the changes.

## 5. Conversion scripts

#### 1) LAS -> PLY
```bash
python3 las_to_ply.py point_cloud.las point_cloud.ply
```

#### 2) Draco encode
```bash
draco_encoder -point_cloud -i point_cloud.ply -o point_cloud.drc
```

#### 3) Draco decode
```bash
draco_decoder -i point_cloud.drc -o point_cloud_draco_decoded.ply
```

#### 4) Draco decompressed PLY -> LAS
```bash
python3 ply_to_las.py point_cloud_draco_decoded.ply point_cloud.ply.meta.json point_cloud_draco_decoded.las
```

#### 5) G-PCC encode
```bash
$TMC3 --mode=0 \
     --uncompressedDataPath=point_cloud.ply \
     --compressedStreamPath=point_cloud.bin \
     --positionQuantizationScale=1 \
     --attribute=color
```

#### 6) G-PCC decode
```bash
$TMC3 --mode=1 \
     --compressedStreamPath=point_cloud.bin \
     --reconstructedDataPath=point_cloud_gpcc_decoded.ply
```

#### 7) G-PCC decompressed PLY -> LAS
```bash
python3 ply_to_las.py point_cloud_gpcc_decoded.ply point_cloud.ply.meta.json point_cloud_gpcc.las
```

