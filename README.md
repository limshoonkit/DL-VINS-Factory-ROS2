# DL-VINS Factory ROS2


## Demo

1. ![NTU-VIRAL combined](media/ntu_combined.mp4)
   
https://github.com/user-attachments/assets/bbd4ba3f-6376-4c9c-a9f7-bf9824ee06c0

2. ![EuRoC combined](media/euroc_combined.mp4)
   
https://github.com/user-attachments/assets/b1bf0218-9f3f-4836-97f0-54fa2f88fc5c

3. ALIKED_LK SubT_MRS smoke room

https://github.com/user-attachments/assets/d04eb0d0-6a45-4aa6-8a83-ee4e06d12958

4. SuperPoint_LightGlue SubT_MRS low_light1

https://github.com/user-attachments/assets/91dc3863-0ca3-4fba-b527-1fde5a592d41

---

## A. Getting Started

```bash
git clone https://github.com/limshoonkit/DL-VINS-Factory-ROS2.git --recursive
```

### Environment

- CUDA 12.6 (Jetpack 6.x)
- TensorRT 10.3
- Ceres 2.2
- OpenCV 4.13

### Engine Export

1. Download weights with 
```bash
chmod +x scripts/download_weights.sh
./scripts/download_weights.sh
```

2. Follow [instructions](./LightGlue-ONNX-Jetson/README.md) to setup uv. Make sure to get polygraphy and onnxsim.

3. Run the [notebooks](./LightGlue-ONNX-Jetson/notebooks). Remember to adjust the batch size and dimensions.

4. Migrate the exported engines to [weights folder](./DL-VINS/src/dl_vins/weights/README.md). Setup format accordingly.

5. For loop-fusion, the vlad master codebook .bin is provided. You only need the DINOv2 ViT-S engine. Export with the provided [script](./AnyLoc/vpr/dino_vpr_export.py). Setup the uv env for Anyloc before that or just trt export from the .onnx provided.

```
cd /home/nvidia/dl-vins-factory/AnyLoc
uv sync
source ./venv/bin/activate
export PATH="/usr/src/tensorrt/bin:$PATH" # for jetson
uv pip install onnx # if missing
uv run python -m vpr.dino_vpr_export \
    --input-h 280 --input-w 448 --clusters 32 --fp16
```

### Build

```bash
cd DL-VINS && colcon build --base-paths src/ --symlink-install \
  --cmake-args=-DCMAKE_BUILD_TYPE=Release --allow-overriding cv_bridge image_geometry
cd ..
```

## B. Running evaluation on public dataset

The script (`scripts/eval_run.sh` + `scripts/eval_runner.py`) run for different public datasets. 

Refer [dataset source](./dataset/README.md) to download it.

Ground truth is pre-baked as `<dataset>/groundtruth/<seq>_gt.tum` and the ROS1 bags are converted to ROS2 .mcap format. Let me know if you need them for those that I am allowed to redistribute based on their license. 

Trajectories are written to `tmp/<dataset>_eval_<mode>/<method>/<seq>/vio_trajectory_run01.tum`.

### Example Quick Run — CPU GFTT only (no engines)

```bash
EVAL_METHOD_SET=dl \
 EVAL_MODE=mono \
 EVAL_USE_LOOP_FUSION=true \
 EVAL_METHODS="dlvins_gftt_cpu_mono_loop" \
 EVAL_DATASET=euroc \
 EVAL_SEQUENCES="MH_01_easy" \
 bash scripts/eval_run.sh
```

### Full run — DL front-ends

Run the full DL sweep for all sequences on a dataset (very long running):

```bash
EVAL_METHOD_SET=dl \
 EVAL_MODE=stereo \
 EVAL_DATASET=euroc \
 bash scripts/eval_run.sh
```

## C. Running with launch file and RViz
In one terminal, run

```
cd DL-VINS
source install/setup.bash
ros2 launch dl_vins ntu_viral_stereo.launch.py \
    extractor:=aliked_lightglue \
    use_loop_fusion:=true \
    rviz:=true

# or 
ros2 launch dl_vins subt_mrs_mono.launch.py \
    extractor:=superpoint_lightglue \
    use_loop_fusion:=true \
    rviz:=true
```

and in another terminal, run
```
source install/setup.bash
ros2 bag play ../dataset/7_NTU-VIRAL/eee_01 --clock

# or
ros2 bag play ../dataset/6_SubT-MRS/low_light1 --clock
```

## D. License / Credits / Acknowledgements
- GPLv3 license

- This work is based off [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) and [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)

- Baseline method for comparison (LET-NET, vins-fusion, okvis2) is available at [baseline_reference](./baseline_reference)

## E. Known Issue

*Feel free to open any Github issues.*

Following are some to beware of

1. **ROS2 QOS profile and DDS discovery**
```
I am using cyclonedds with profile available at ./scripts/cyclonedds.xml. This increase the buffer size. Also for sensor qos, its default is reliable. The evaluation script also set a non-default ROS_DOMAIN_ID.

I have found that intermittent network connection, ros multicast and some bad actors spoofing around can mess up the long running evaluation. My advice, run evaluation offline. Especially the bad actor part, took me a week to realise this.
```

2. **Jetson Thermal Issue**
```
Basically GPU uptime is very long considering the length of evaluation data. Make sure to keep it in a cool place. I am also using an external SSD for the dataset, those get really hot as well while running.
```

3. **Non-determinism**
```
GPU, multithread, different platform architecture, TensorRT version, floating point arithmetic, RANSAC differnt seed just to name a few. Variance should be small, file an issue if result is completely different than what I reported.
```
