# UoSM-Campus — handheld LiDAR-visual-inertial dataset

Nine handheld sequences recorded around the University of Southampton Malaysia
campus, day and night, with a Livox Mid-360 + ZED 2i + HikRobot camera rig.
~1.9 km of total trajectory.

---

## Layout

```
1_UoSM-Campus/
├── <SEQ>/
│   ├── rosbag_<timestamp>/          ROS 2 mcap — all sensors, host clock
│   └── svo_<timestamp>/             ZED .svo2 — camera-native clock
├── LF-02-N/
│   └── rosbag_svo_reexport/         see "LF-02-N" below
├── ground_truth/                    TUM format (.txt)
│   ├── fastlivo2/<SEQ>.txt          FAST-LIVO2 odometry only
│   └── fastlivo2_lvba_blended/<SEQ>.txt   FAST-LIVO2 + Global-LVBA (use this)
├── FastCalib-HikCam_Mid360/         HikRobot camera <-> Mid-360 extrinsics
└── kalibr_SLZed_HikCam/             ZED <-> HikRobot Kalibr calibration
```

### Sequences

`BF` = both floor, `LF` = lower floor, `UF` = upper floor; `-D` = day, `-N` = night.

`Path` is distance travelled; `Span` is the diagonal of the trajectory's
axis-aligned bounding box, i.e. how far the sequence spreads spatially. A sequence can have a large path but small span.

| Sequence | Poses | Duration [s] | Path [m] | Span [m] |
|---|---|---|---|---|
| BF-00   | 2898 | 416.3 | 507.4 | 89.9 |
| LF-01-D | 1240 | 168.0 | 184.3 | 48.4 |
| LF-01-N | 1397 | 189.3 | 183.5 | 54.6 |
| LF-02-D | 1037 | 145.1 | 158.3 | 64.1 |
| LF-02-N |  912 | 129.8 | 141.6 | 57.8 |
| UF-01-D |  999 | 140.7 | 162.9 | 67.2 |
| UF-01-N |  965 | 137.4 | 152.0 | 66.0 |
| UF-02-D | 1217 | 175.9 | 204.1 | 65.2 |
| UF-02-N | 1228 | 180.2 | 198.4 | 69.0 |


### Topics

| Topic | Type | Rate |
|---|---|---|
| `/zed_node/left/color/rect/image`  | `sensor_msgs/msg/Image` | 30 Hz |
| `/zed_node/right/color/rect/image` | `sensor_msgs/msg/Image` | 30 Hz |
| `/zed_node/imu/data`               | `sensor_msgs/msg/Imu` | 202 Hz |
| `/livox/lidar`                     | `sensor_msgs/msg/PointCloud2` | 10 Hz |
| `/livox/imu`                       | `sensor_msgs/msg/Imu` | 200 Hz |
| `/left_camera/image`               | `sensor_msgs/msg/Image` | 10 Hz |

---

## Ground truth (TUM format)

Both sets are plain-text **TUM trajectory format**: one pose per line, space
separated, no header —

```
timestamp tx ty tz qx qy qz qw
1590192263.400343 -0.035121 0.080318 0.018658 0.007820 0.222244 -0.002448 0.974957
```

Translation in metres, rotation as a unit quaternion (x, y, z, **w** last).
Timestamps are on the LiDAR clock, one pose per LiDAR scan (~10 Hz). Directly
loadable by `evo` (`evo_ape tum ...`), and by anything that reads TUM.

- **`ground_truth/fastlivo2/`** — raw FAST-LIVO2 LiDAR-visual-inertial odometry.
  Locally accurate, but drifts over a full loop.
- **`ground_truth/fastlivo2_lvba_blended/`** — FAST-LIVO2 refined by Global-LVBA
  (global LiDAR-visual bundle adjustment) and blended back onto the FAST-LIVO2
  timeline. **This is the reference to evaluate against.** It is a *pseudo* ground
  truth: no external motion capture or RTK, so treat it as accurate to roughly the
  the centimetre, not millimetre.

The blended file has slightly fewer poses than the FAST-LIVO2 one (e.g. 912 vs 925
on LF-02-N) — the ends are trimmed where the BA window has no support.

---

## LF-02-N — the ZED dropout, and how to re-export

**The problem.** The original mcap for LF-02-N is missing **2.86 s of the entire
ZED stream** at t≈53 s — both images and the ZED IMU stop together, while the Livox
stream continues. Max image gap is 2860 ms against a nominal 33 ms. Any VIO run on
this bag has to coast blind through nearly three seconds.

**The fix.** Re-exporting the ZED stream from the `.svo2`
recovers the missing data. `rosbag_svo_reexport/` is that re-export. It carries only
the three ZED topics, which is the complete stereo-inertial VIO input. The Livox
stream (and therefore the ground truth) still comes from the original mcap.

### How to re-export

The export opens the `.svo2` with the ZED SDK, walks it frame by frame, and writes
`/zed_node/{left,right}/color/rect/image` plus `/zed_node/imu/data` into an mcap at
SDK timestamps. Two details are essential:

1. **Coordinate system must be `COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD`.**
   This is the single easiest thing to get wrong, and it silently destroys the bag:
   - The SDK default is `IMAGE`, which puts gravity on **−Y** instead of +Z.
   - `RIGHT_HANDED_Z_UP` fixes gravity to +Z but is **still yawed 90° from ROS**
     (X right, not X forward), so accel and gyro x/y come out swapped and negated.

  The imu is also at ~400 Hz rather than ~200Hz.

2. **Angular velocity needs `deg2rad`.** The SDK reports deg/s; the ROS wrapper
   publishes rad/s.

---

## Benchmark results

ATE (APE RMSE, m) against `ground_truth/fastlivo2_lvba_blended/`, SE(3)-Umeyama
aligned, per-sequence clock offset fitted by speed-profile cross-correlation.

DL-VINS front-ends from <https://github.com/limshoonkit/DL-VINS-Factory-ROS2>, run
stereo-inertial. Scored stream is `/dl_vins/odometry` (no loop-closure), so these
are pure odometry numbers. ZED GEN_3 is the ZED SDK's own onboard VIO at `GEN_3`
positional-tracking mode, for reference.

| Method | BF-00 | LF-01-D | LF-01-N | LF-02-D | LF-02-N | UF-01-D | UF-01-N | UF-02-D | UF-02-N | median |
|---|---|---|---|---|---|---|---|---|---|---|
| **ZED GEN_3** | 1.43 | **0.38** | 0.51 | 0.69 | 0.84 | 2.38 | 0.72 | 2.28 | **0.47** | **0.72** |
| ALIKED + LightGlue    | 1.37 | 0.49 | 0.35 | 0.73 | 0.57 | 1.01 | 0.94 | 0.99 | 0.58 | 0.73 |
| ALIKED + LK           | 1.12 | 0.62 | 0.69 | 0.39 | 0.98 | 1.01 | 3.28 | 1.76 | 1.57 | 1.01 |
| SuperPoint + LightGlue| 1.82 | 0.59 | **0.30** | 0.99 | **0.42** | 1.09 | 1.48 | 1.12 | 0.69 | 0.99 |
| SuperPoint + LK       | 1.15 | 0.40 | 0.93 | 0.83 | 0.64 | 1.26 | 2.55 | 2.24 | 2.09 | 1.15 |
| XFeat + LightGlue     | 1.33 | 0.51 | 0.88 | 0.95 | 0.98 | 2.08 | 3.29 | 1.66 | 1.16 | 1.16 |
| XFeat + LK            | 1.71 | 0.54 | 0.92 | 0.59 | 1.05 | 1.03 | 2.68 | 2.03 | 2.53 | 1.05 |
| **RaCo + LightGlue**  | 2.22 | 0.46 | 0.40 | **0.39** | 0.66 | 1.12 | **0.63** | **0.94** | 0.64 | **0.64** |
| RaCo + LK             | 1.30 | 0.57 | 0.79 | 0.37 | 1.14 | 1.38 | 2.79 | 2.14 | 1.09 | 1.14 |
| GFTT (classical)      | 1.61 | 0.53 | 0.43 | 0.53 | 0.78 | 0.83 | 2.34 | 1.83 | 0.78 | 0.78 |

Summary over all 9 sequences:

| Method | median | mean | worst | x | y | z |
|---|---|---|---|---|---|---|
| RaCo + LightGlue | **0.64** | 0.83 | 2.22 | 0.62 | 0.49 | 0.15 |
| ZED GEN_3 | 0.72 | 1.08 | 2.38 | 0.64 | 0.60 | 0.54 |
| ALIKED + LightGlue | 0.73 | **0.78** | **1.37** | 0.59 | 0.44 | 0.17 |
| GFTT | 0.78 | 1.07 | 2.34 | 0.77 | 0.66 | 0.19 |
| SuperPoint + LightGlue | 0.99 | 0.94 | 1.82 | 0.76 | 0.51 | 0.15 |
| ALIKED + LK | 1.01 | 1.27 | 3.28 | 0.97 | 0.71 | 0.25 |
| XFeat + LK | 1.05 | 1.45 | 2.68 | 1.11 | 0.82 | 0.28 |
| RaCo + LK | 1.14 | 1.29 | 2.79 | 0.95 | 0.74 | 0.25 |
| SuperPoint + LK | 1.15 | 1.34 | 2.55 | 1.02 | 0.76 | 0.26 |
| XFeat + LightGlue | 1.16 | 1.43 | 3.29 | 1.09 | 0.82 | 0.16 |

---

## Known issues

- **LF-02-N**: 2.86 s ZED dropout in the mcap. Use `rosbag_svo_reexport/` for any
  ZED-based VIO. The Livox stream in the original bag is unaffected.
- The mcap record-clock epoch is **per recording session, not global** — do not
  reuse a time offset fitted on one sequence for another. Values are tabulated
  below.

### Clock offsets

The bag carries two independent clocks. The ZED topics are stamped on the **host**
clock; `/livox/*` is stamped on the **Livox** clock, and the ground truth is built
from the Livox stream, so **the ground truth is on the Livox clock**. To compare a
ZED-based trajectory against the ground truth, subtract Δ:

```
t_livox = t_host − Δ
```

| Sequence | Δ [s] | Recording session |
|---|---|---|
| BF-00   | 176718469.580 | 2025-12-28 (BF) |
| LF-01-D | 176708459.435 | 2025-12-28 (day field) |
| LF-02-D | 176708459.426 | 2025-12-28 (day field) |
| UF-01-D | 176708459.412 | 2025-12-28 (day field) |
| UF-02-D | 176708459.416 | 2025-12-28 (day field) |
| LF-01-N | 176650851.002 | 2025-12-27 (LF night) |
| UF-01-N | 176648552.016 | 2025-12-27 (UF night) |
| UF-02-N | 176648552.008 | 2025-12-27 (UF night) |
| LF-02-N | 178551189.897 | 2026-01-18 |

Derived as `median(log_time − header.stamp)` over `/livox/imu` minus the same over
`/zed_node/imu/data`, so the shared record latency cancels and what remains is the
epoch difference. Measured over an 8 s window at each end of every bag, Δ is
**constant to within ±4 ms** (drift < 30 µs/s), so a single constant per sequence is
sufficient — no linear drift term needed.

Note the grouping: the four day sequences share one session and agree to 23 ms;
the two UF night sequences agree to 8 ms; LF-02-N, recorded three weeks later, is
1.84e6 s away. That is why a single global constant cannot work. 


**The LF-02-N re-export is on a third clock** — the ZED SDK's camera clock, not the
host clock — so Δ above does not apply to `rosbag_svo_reexport/`. Fit that one
directly (the offset is constant, drift < 39 µs/s over 128 s); most evaluation
pipelines that cross-correlate speed profiles will pick it up automatically. 
The original LF-02-N on 2025-12-27 has to be discarded and re-recorded three weeks later because that sequence contain many humans (dynamic objects) walking around the scene which corrupts the overall ground truth.