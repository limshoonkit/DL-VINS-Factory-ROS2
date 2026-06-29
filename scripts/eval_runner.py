"""
Multi-dataset DL-VINS evaluation runner

Env overrides (set by eval_run.sh)
    EVAL_DATASET          "euroc" | "ntu_viral" | "botanic_garden" | "kaist_vio"  (default: "euroc")
    EVAL_MODE             "mono" | "stereo"           (default: "mono")
    EVAL_RUN_TAG          arbitrary suffix on RESULTS_DIR  (default: "")
    EVAL_USE_LOOP_FUSION  "true"|"false"              (default: "false")
    EVAL_RESULTS_DIR      override full results dir   (default: derived from above)
    EVAL_WORKSPACE_DIR    dl-vins workspace           (default: ~/dl-vins-factory/DL-VINS)
    EVAL_VINS_FUSION_DIR  vins-fusion workspace       (default: ~/vins-fusion)
    EVAL_DATASET_DIR      bag root override           (default: DATASETS[EVAL_DATASET]['dataset_dir'])
    EVAL_GT_DIR           GT root override            (default: DATASETS[EVAL_DATASET]['gt_dir'])
    EVAL_ROS_DOMAIN_ID    explicit ROS_DOMAIN_ID      (default: 69)
    EVAL_METHOD_SET       "classic" | "dl"            (default: "classic")
    EVAL_OKVIS2_SETUP     okvis2 colcon overlay setup.bash

CLI (single-trial worker):
    python3 eval_runner.py --method <label> --sequence <seq> --trial <n>
    python3 eval_runner.py --list
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil


# ─── ENV-DRIVEN CONFIGURATION ─────────────────────────────────────────────────

DATASET: str = os.environ.get("EVAL_DATASET", "euroc")
MODE: str = os.environ.get("EVAL_MODE", "mono")
RUN_TAG: str = os.environ.get("EVAL_RUN_TAG", "")
USE_LOOP_FUSION: bool = os.environ.get("EVAL_USE_LOOP_FUSION", "false").lower() == "true"

WORKSPACE_DIR: str = os.environ.get(
    "EVAL_WORKSPACE_DIR", os.path.join(_REPO_ROOT, "DL-VINS"))
VINS_FUSION_DIR: str = os.environ.get(
    "EVAL_VINS_FUSION_DIR", os.path.join(_REPO_ROOT, "baseline_reference/vins-fusion_ws"))
OKVIS2_SETUP: str = os.environ.get(
    "EVAL_OKVIS2_SETUP", os.path.join(_REPO_ROOT, "baseline_reference/okvis2_ws/install/setup.bash"))

SCRIPTS_DIR: str = str(Path(__file__).resolve().parent)
_REPO_ROOT: str  = str(Path(__file__).resolve().parents[1])
# Datasets now live under <repo>/dataset/<name> (moved off the external SSD).
DATASET_ROOT: str = os.environ.get(
    "EVAL_DATASET_ROOT", str(Path(__file__).resolve().parents[1] / "dataset"))
OKVIS2_CONFIG_DIR: str = os.environ.get(
    "EVAL_OKVIS2_CONFIG_DIR", os.path.join(SCRIPTS_DIR, "okvis2_configs"))
VF_CONFIG_DIR: str = os.environ.get(
    "EVAL_VF_CONFIG_DIR", os.path.join(SCRIPTS_DIR, "vins_fusion_configs"))

METHOD_SET: str = os.environ.get("EVAL_METHOD_SET", "classic")

# Per-dataset/per-mode config file names (relative to VF_CONFIG_DIR / OKVIS2_CONFIG_DIR).
VF_CONFIG: dict[str, dict[str, str | None]] = {
    "euroc":          {"mono": "euroc/euroc_mono_imu_config.yaml",
                       "stereo": "euroc/euroc_stereo_imu_config.yaml"},
    "ntu_viral":      {"mono": "ntu_viral/viral_mono_imu_config.yaml",
                       "stereo": "ntu_viral/viral_stereo_imu_config.yaml"},
    "botanic_garden": {"mono": "botanic_garden/botanic_mono_imu_config.yaml",
                       "stereo": "botanic_garden/botanic_stereo_imu_config.yaml"},
    "botanic_garden_rgb": {"mono": "botanic_garden/botanic_color_mono_imu_config.yaml",
                       "stereo": "botanic_garden/botanic_color_stereo_imu_config.yaml"},
    "kaist_vio":      {"mono": "kaist_vio/kaist_mono_imu_config.yaml",
                       "stereo": "kaist_vio/kaist_stereo_imu_config.yaml"},
    "subt_mrs":       {"mono": "__seq__", "stereo": None},
}
OKVIS_CONFIG: dict[str, dict[str, str | None]] = {
    "euroc":          {"mono": "euroc_mono.yaml",          "stereo": "euroc_stereo.yaml"},
    "ntu_viral":      {"mono": "ntu_viral_mono.yaml",      "stereo": "ntu_viral_stereo.yaml"},
    "botanic_garden": {"mono": "botanic_garden_mono.yaml", "stereo": "botanic_garden_stereo.yaml"},
    "botanic_garden_rgb": {"mono": "botanic_garden_rgb_mono.yaml", "stereo": "botanic_garden_rgb_stereo.yaml"},
    "kaist_vio":      {"mono": "kaist_vio_mono.yaml",      "stereo": "kaist_vio_stereo.yaml"},
}
# Bag image/imu topics per dataset; OKVIS2's RosbagReader/Subscriber expect the fixed name so we remap topics
OKVIS_REMAP: dict[str, dict[str, str]] = {
    "euroc":          {"img0": "/cam0/image_raw",            "img1": "/cam1/image_raw",             "imu": "/imu0"},
    "ntu_viral":      {"img0": "/right/image_raw",           "img1": "/left/image_raw",             "imu": "/imu/imu"},
    "botanic_garden": {"img0": "/dalsa_gray/left/image_raw", "img1": "/dalsa_gray/right/image_raw", "imu": "/imu/data"},
    "botanic_garden_rgb": {"img0": "/dalsa_color/left/image_raw", "img1": "/dalsa_color/right/image_raw", "imu": "/imu/data"},
    "kaist_vio":      {"img0": "/camera/infra1/image_rect_raw", "img1": "/camera/infra2/image_rect_raw", "imu": "/mavros/imu/data"},
}

# Per-dataset registry. Each entry describes how to locate bags + GT and which
# launch file family to use.
DATASETS: dict[str, dict] = {
    "euroc": {
        "dataset_dir":   os.path.join(DATASET_ROOT, "2_EuRoC"),
        "gt_dir":        os.path.join(DATASET_ROOT, "2_EuRoC", "groundtruth"),
        "bag_pattern":   "{seq}/{seq}.db3",         # marker file inside bag dir
        "bag_play_dir":  "{seq}",                   # what `ros2 bag play` receives
        "launch_prefix": "euroc",
        "sequences": [
            "MH_01_easy","MH_02_easy","MH_03_medium","MH_04_difficult","MH_05_difficult",
            "V1_01_easy","V1_02_medium","V1_03_difficult",
            "V2_01_easy","V2_02_medium","V2_03_difficult",
        ],
    },
    "ntu_viral": {
        "dataset_dir":   os.path.join(DATASET_ROOT, "7_NTU-VIRAL"),
        "gt_dir":        os.path.join(DATASET_ROOT, "7_NTU-VIRAL", "groundtruth"),
        "bag_pattern":   "{seq}/{seq}/{seq}.mcap",   # nested
        "bag_play_dir":  "{seq}/{seq}",
        "launch_prefix": "ntu_viral",
        "sequences": [
            "eee_01","eee_02","eee_03",
            "nya_01","nya_02","nya_03",
            "rtp_01","rtp_02","rtp_03",
            "sbs_01","sbs_02","sbs_03",
            "spms_01","spms_02","spms_03",
            "tnp_01","tnp_02","tnp_03",
        ],
    },
    "botanic_garden": {
        "dataset_dir":   os.path.join(DATASET_ROOT, "8_Botanic-Garden"),
        "gt_dir":        os.path.join(DATASET_ROOT, "8_Botanic-Garden", "groundtruth"),
        "bag_pattern":   "{seq}_VLIO/{seq}_VLIO.mcap",
        "bag_play_dir":  "{seq}_VLIO",
        "launch_prefix": "botanic_garden",
        "sequences": [
            "1005_00","1005_01","1005_07","1006_01","1008_03","1018_00","1018_13",
        ],
    },
    "botanic_garden_rgb": {
        "dataset_dir":   os.path.join(DATASET_ROOT, "8_Botanic-Garden"),
        "gt_dir":        os.path.join(DATASET_ROOT, "8_Botanic-Garden", "groundtruth"),
        "bag_pattern":   "{seq}_VLIO/{seq}_VLIO.mcap",
        "bag_play_dir":  "{seq}_VLIO",
        "launch_prefix": "botanic_garden_rgb",
        "sequences": [
            # NOTE: 1005_00 and 1008_03 have color (RGB) at a lower/uneven rate
            # than gray (~3.9 Hz and ~7.4 Hz vs ~9 Hz); gray stereo is full-rate.
            "1005_00","1005_01","1005_07","1006_01","1008_03","1018_00","1018_13",
        ],
    },
    "kaist_vio": {
        "dataset_dir":   os.path.join(DATASET_ROOT, "3_Kaist-VIO"),
        "gt_dir":        os.path.join(DATASET_ROOT, "3_Kaist-VIO", "groundtruth"),
        "bag_pattern":   "{seq}/{seq}.mcap",
        "bag_play_dir":  "{seq}",
        "launch_prefix": "kaist_vio",
        "sequences": [
            "circle","circle_fast","circle_head",
            "infinite","infinite_fast","infinite_head",
            "rotation","rotation_fast",
            "square","square_fast","square_head",
        ],
    },
    "subt_mrs": {
        "dataset_dir":    os.path.join(DATASET_ROOT, "6_SubT-MRS"),
        "gt_dir":         os.path.join(DATASET_ROOT, "6_SubT-MRS", "groundtruth"),
        "bag_pattern":    "{seq}/{seq}.mcap",
        "bag_play_dir":   "{seq}",
        "launch_prefix":  "subt_mrs",
        # Sequences have per-sequence calibration; the launch file needs seq:={seq}.
        "seq_in_launch":  True,
        # Visual-track sequences — all on the RC7 sensor rig (shared calib/config).
        "seq_vf_configs": {
            "flash_light":    "subt_mrs/subt_rc7_mono_imu_config.yaml",
            "low_light1":     "subt_mrs/subt_rc7_mono_imu_config.yaml",
            "low_light2":     "subt_mrs/subt_rc7_mono_imu_config.yaml",
            "over_exposure":  "subt_mrs/subt_rc7_mono_imu_config.yaml",
            "smoke_handheld": "subt_mrs/subt_rc7_mono_imu_config.yaml",
        },
        "sequences": [
            "flash_light",
            "low_light1",
            "low_light2",
            "over_exposure",
            "smoke_handheld",
        ],
    },
}

if DATASET not in DATASETS:
    raise SystemExit(
        f"EVAL_DATASET={DATASET!r} unknown — choose one of {list(DATASETS)}")

_DS = DATASETS[DATASET]
DATASET_DIR: str = os.environ.get("EVAL_DATASET_DIR", _DS["dataset_dir"])
GT_DIR:      str = os.environ.get("EVAL_GT_DIR",      _DS["gt_dir"])

_loop_tag = "_loop" if USE_LOOP_FUSION else ""
RESULTS_DIR: str = os.environ.get(
    "EVAL_RESULTS_DIR",
    os.path.join(_REPO_ROOT, f"tmp/{DATASET}_eval_{MODE}{_loop_tag}{RUN_TAG}"))


ROS_DOMAIN_ID: int = int(os.environ.get("EVAL_ROS_DOMAIN_ID", "69"))
os.environ["ROS_DOMAIN_ID"] = str(ROS_DOMAIN_ID)

SEQUENCES: list[str] = list(_DS["sequences"])

NUM_TRIALS: int = 1
BAG_RATE: float = 1.0
BAG_TIMEOUT_SLOWDOWN: float = 3.0
SETTLE_TIME: int = 6
IMAGE_GATE_TIMEOUT: float = 20.0

COLORS: dict[str, str] = {
    "vins_fusion":                   "#222222",
    "okvis2_slam":                   "#1f9e89",
    "okvis2_vio":                    "#86c5bb",
    "dlvins_gftt_cpu":               "#4c72b0",
    "dlvins_aliked_lk":              "#dd8452",
    "dlvins_aliked_lightglue":       "#cc4c1c",
    "dlvins_superpoint_lk":          "#55a868",
    "dlvins_superpoint_lightglue":   "#2e8a3a",
    "dlvins_raco_lk":                "#c44e52",
    "dlvins_raco_lightglue":         "#8c1a1d",
    "dlvins_xfeat_lk":               "#8172b2",
    "dlvins_xfeat_lightglue":        "#5a3f87",
    "dlvins_sift_cpu_lk":            "#bcbd22",
    "dlvins_sift_cpu_lightglue":     "#8a8f0e",
}

# DL front-end sweep for METHOD_SET="dl"
DL_EXTRACTORS_LG = [
    "aliked_lightglue", "superpoint_lightglue", "raco_lightglue", "xfeat_lightglue",
    # "sift_cpu_lightglue",
]
# DL_EXTRACTORS_LK = ["aliked_lk", "superpoint_lk", "raco_lk", "xfeat_lk", "sift_cpu_lk"]
DL_EXTRACTORS_LK = ["aliked_lk", "superpoint_lk", "raco_lk", "xfeat_lk"]


def _tracker_method(extractor: str) -> str:
    """Map a launch extractor name to the CSV method tag emitted by metrics_logger."""
    if extractor.startswith("aliked"):     return "aliked"
    if extractor.startswith("superpoint"): return "superpoint"
    if extractor.startswith("raco"):       return "raco"
    if extractor.startswith("xfeat"):      return "xfeat"
    if extractor.startswith("sift"):       return "sift_cpu"
    if extractor.startswith("gftt"):       return "gftt_cpu"
    return extractor


def _build_compare_methods(mode: str) -> list[dict]:
    """Classic comparison: OKVIS2 vs VINS-Fusion.

    With loop closure on, base (no-PGO) and corrected trajectories are captured
    per method: VINS-Fusion via dual topic recording (raw + rect),
    OKVIS2 via two runs (vio = do_loop_closures:false, slam = true).
    """
    assert mode in ("mono", "stereo"), mode
    loop = USE_LOOP_FUSION
    _lt = "_loop" if loop else ""
    methods: list[dict] = []

    # ── VINS-Fusion (vins_node + optional loop_fusion_node) ──
    # "__seq__" sentinel means config is resolved per-sequence in run_sequence.
    _vf_cfg_rel = VF_CONFIG.get(DATASET, {}).get(mode)
    if _vf_cfg_rel is not None:
        _vf_cfg_abs = (None if _vf_cfg_rel == "__seq__"
                       else os.path.join(VF_CONFIG_DIR, _vf_cfg_rel))
        methods.append({
            "label":          f"vins_fusion_{mode}{_lt}",
            "runner":         "vins_fusion",
            "config":         _vf_cfg_abs,   # None → resolved per-sequence in run_sequence
            "odom_topic":     "/odometry_rect" if loop else "/vins_estimator/odometry",
            "odom_msg_type":  "odometry",
            "raw_odom_topic": "/vins_estimator/odometry" if loop else None,
            "has_tracker":    False,
            "has_estimator":  False,
            "use_loop_fusion": loop,
            "color":          COLORS.get("vins_fusion", "#222222"),
        })

    # ── OKVIS2 (okvis_node_subscriber; bag topics remapped to /okvis/*) ──
    _okvis_cfg_rel = OKVIS_CONFIG.get(DATASET, {}).get(mode)
    if _okvis_cfg_rel is not None:
        _okvis_is_seq = (_okvis_cfg_rel == "__seq__")
        if not _okvis_is_seq:
            okvis_cfg = os.path.join(OKVIS2_CONFIG_DIR, _okvis_cfg_rel)
        else:
            okvis_cfg = None   # resolved per-sequence in run_sequence
        rm = OKVIS_REMAP[DATASET]
        bag_remap = [f"{rm['img0']}:=/okvis/cam0/image_raw", f"{rm['imu']}:=/okvis/imu0"]
        if mode == "stereo":
            bag_remap.append(f"{rm['img1']}:=/okvis/cam1/image_raw")
    # slam = loop-corrected, vio = base (no pose-graph loop closures). The okvis
    # node writes the optimised keyframe trajectory to CSV (rpg/TUM) which we read
    # directly instead of recording a topic.
    for variant, do_loop in (("slam", True), ("vio", False)):
        if _okvis_cfg_rel is None:
            break
        methods.append({
            "label":          f"okvis2_{mode}_{variant}",
            "runner":         "okvis2",
            "okvis_config":   okvis_cfg,
            "okvis_do_loop":  do_loop,
            "okvis_mode":     variant,
            "bag_remap":      bag_remap,
            "odom_topic":     None,        # read CSV instead of recording a topic
            "odom_msg_type":  "odometry",
            "raw_odom_topic": None,
            "has_tracker":    False,
            "has_estimator":  False,
            "use_loop_fusion": False,
            "color":          COLORS.get(f"okvis2_{variant}", "#1f9e89"),
        })
    return methods


def _build_methods(mode: str) -> list[dict]:
    if METHOD_SET == "classic":
        return _build_compare_methods(mode)
    assert mode in ("mono", "stereo"), mode
    launch_file = f"{_DS['launch_prefix']}_{mode}.launch.py"

    if mode == "mono":
        extractors = ["gftt_cpu"] + DL_EXTRACTORS_LG + DL_EXTRACTORS_LK
    else:
        extractors = ["gftt_cpu"] + DL_EXTRACTORS_LG + DL_EXTRACTORS_LK

    methods: list[dict] = []

    for ext in extractors:
        # Loop fusion wires for every front-end (mono and stereo): gftt/raco ->
        # BRIEF loop closure, aliked/superpoint/xfeat -> local_agg DL loop closure.
        # traj = /pose_graph_path (pgo odom); traj_raw = /dl_vins/odometry (dl odom).
        wire_loop = bool(USE_LOOP_FUSION)
        odom_topic = "/pose_graph_path" if wire_loop else "/dl_vins/odometry"
        odom_msg_type = "path" if wire_loop else "odometry"

        methods.append({
            "label":         f"dlvins_{ext}_{mode}{'_loop' if wire_loop else ''}",
            "runner":        "dl_vins",
            "launch_file":   launch_file,
            "extractor":     ext,
            "tracker_method": _tracker_method(ext),
            "odom_topic":    odom_topic,
            "odom_msg_type": odom_msg_type,
            "raw_odom_topic": "/dl_vins/odometry" if wire_loop else None,
            "has_tracker":   True,
            "has_estimator": True,
            "use_loop_fusion": wire_loop,
            "color":         COLORS.get(f"dlvins_{ext}", "#4c72b0"),
        })
    return methods


METHODS: list[dict] = _build_methods(MODE)


# ─── PROCESS / RECORDER HELPERS ───────────────────────────────────────────────

def find_ros_node_pid(parent_pid: int, node_name: str,
                      timeout: float = 15.0) -> int | None:
    """Wait for a child process whose name contains ``node_name`` to spawn."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            parent = psutil.Process(parent_pid)
            for child in parent.children(recursive=True):
                if node_name in child.name():
                    return child.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        time.sleep(0.5)
    return None


def _image_gate_topic(method: dict) -> str | None:
    """Image topic on which the bag's publisher and the VIO node's subscription
    must rendezvous before playback starts."""
    if method.get('runner') != 'dl_vins':
        return None
    rm = OKVIS_REMAP.get(DATASET)
    return rm["img0"] if rm else None


def topic_endpoint_counts(setup_cmd: str, topic: str) -> tuple[int, int]:
    """(publisher_count, subscription_count) for ``topic`` via ros2 topic info -v.
    (-1, -1) if the query fails (treated as 'not yet matched')."""
    try:
        out = subprocess.run(
            f"bash -c '{setup_cmd} && ros2 topic info {topic} --verbose'",
            shell=True, capture_output=True, text=True, timeout=10).stdout
    except subprocess.SubprocessError:
        return (-1, -1)
    pub = sub = -1
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Publisher count:"):
            pub = int(s.split(":", 1)[1])
        elif s.startswith("Subscription count:"):
            sub = int(s.split(":", 1)[1])
    return (pub, sub)


def wait_for_pub_sub_match(setup_cmd: str, topic: str, timeout: float) -> bool:
    """Poll until ``topic`` has >=1 publisher AND >=1 subscription, i.e. the bag's
    (paused) publisher and the node's subscription are both in the graph. Returns
    True on match, False on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pub, sub = topic_endpoint_counts(setup_cmd, topic)
        if pub >= 1 and sub >= 1:
            return True
        time.sleep(0.5)
    return False


def record_trajectory(setup_cmd: str, workspace_dir: str,
                      traj_path: str, odom_topic: str,
                      stderr_path: str | None = None,
                      msg_type: str = "odometry") -> subprocess.Popen:
    """Spawn a child Python process that subscribes to ``odom_topic`` and writes TUM.

    msg_type:
      - "odometry": append one TUM row per Odometry callback
      - "path":     overwrite the file with the full Path each callback (last-msg-wins,
                    suited to /pose_graph_path which republishes the entire corrected trace)
    """
    if msg_type == "odometry":
        recorder_script = f"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class TrajRecorder(Node):
    def __init__(self, path):
        super().__init__('traj_recorder')
        self.f = open(path, 'w')
        self.count = 0
        self.sub = self.create_subscription(
            Odometry, '{odom_topic}', self.cb, 10)

    def cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.f.write(f'{{t:.9f}} {{p.x}} {{p.y}} {{p.z}} {{q.x}} {{q.y}} {{q.z}} {{q.w}}\\n')
        self.f.flush()
        self.count += 1

    def destroy_node(self):
        print(f'[traj_recorder] Recorded {{self.count}} poses', flush=True)
        self.f.close()
        super().destroy_node()

rclpy.init()
node = TrajRecorder('{traj_path}')
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
"""
    elif msg_type == "path":
        recorder_script = f"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path

class PathRecorder(Node):
    def __init__(self, path):
        super().__init__('traj_recorder')
        self.path = path
        self.last_n = 0
        self.sub = self.create_subscription(
            Path, '{odom_topic}', self.cb, 10)

    def cb(self, msg):
        with open(self.path, 'w') as f:
            for ps in msg.poses:
                t = ps.header.stamp.sec + ps.header.stamp.nanosec * 1e-9
                p = ps.pose.position
                q = ps.pose.orientation
                f.write(f'{{t:.9f}} {{p.x}} {{p.y}} {{p.z}} {{q.x}} {{q.y}} {{q.z}} {{q.w}}\\n')
        self.last_n = len(msg.poses)

    def destroy_node(self):
        print(f'[traj_recorder] Final Path had {{self.last_n}} poses', flush=True)
        super().destroy_node()

rclpy.init()
node = PathRecorder('{traj_path}')
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
"""
    else:
        raise ValueError(f"unknown msg_type: {msg_type!r}")

    script_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', prefix='traj_recorder_', delete=False)
    script_file.write(recorder_script)
    script_file.close()

    cmd = f"bash -c '{setup_cmd} && python3 {script_file.name}'"
    stderr_target = open(stderr_path, 'w') if stderr_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.DEVNULL, stderr=stderr_target,
        cwd=workspace_dir, start_new_session=True,
    )
    proc._script_file = script_file.name
    proc._stderr_file = stderr_target if stderr_path else None
    return proc


def bag_duration_seconds(bag_dir: str, fallback: float = 600.0) -> float:
    """Read bag duration (s) from rosbag2 metadata.yaml; fall back if missing."""
    meta = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.exists(meta):
        return fallback
    with open(meta) as f:
        for line in f:
            s = line.strip()
            if s.startswith("nanoseconds:"):
                try:
                    return int(s.split(":", 1)[1].strip()) * 1e-9
                except ValueError:
                    return fallback
    return fallback


def find_latest_csv(log_dir: str, prefix: str, after_time: float) -> str | None:
    """Pick the most recent CSV in ``log_dir`` matching ``{prefix}*.csv``."""
    candidates = glob.glob(os.path.join(log_dir, f"{prefix}*.csv"))
    candidates = [c for c in candidates if os.path.getmtime(c) >= after_time]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def kill_proc_group(proc: subprocess.Popen | None,
                    sig=signal.SIGTERM, wait: float = 5.0) -> None:
    """Send `sig` to the Popen's process group; escalate to SIGKILL on timeout."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, OSError):
        return
    try:
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def _pkill_ros_orphans() -> None:
    """SIGKILL any leftover component_container / ros2 launch / bag play / traj_recorder
    processes that escaped the killed process group. Idempotent."""
    for pat in ("component_container", "ros2 launch", "ros2 bag play",
                "traj_recorder", "vins_node", "loop_fusion_node",
                "okvis_node_subscriber"):
        subprocess.run(["pkill", "-9", "-f", pat],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Brief pause so DDS / TCP sockets close before the next trial subscribes.
    time.sleep(0.5)


def kill_vio_cascade(proc: subprocess.Popen | None) -> None:
    """Tear down a VIO launch tree reliably.

    ROS2 launch + composable-node containers don't always honor a single SIGTERM at the
    group level — `component_container_mt` can outlive the launch process and keep
    publishing odometry into the next trial. Escalate SIGINT (graceful rclcpp shutdown)
    → SIGTERM → SIGKILL, then pkill any escapees by process name.
    """
    if proc is None:
        _pkill_ros_orphans()
        return
    kill_proc_group(proc, signal.SIGINT, wait=3.0)
    if proc.poll() is None:
        kill_proc_group(proc, signal.SIGTERM, wait=3.0)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=2.0)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass
    _pkill_ros_orphans()


def _write_okvis_config(src_path: str, dst_path: str, do_loop: bool) -> None:
    """Copy an OKVIS2 config, forcing estimator_parameters.do_loop_closures.

    slam (corrected) -> true, vio (base odometry) -> false. OKVIS reads the flag
    from the YAML (cv::FileStorage), so it can't be a CLI/ROS param override.
    """
    txt = Path(src_path).read_text()
    val = "true" if do_loop else "false"
    txt = txt.replace("do_loop_closures: true",  f"do_loop_closures: {val}")
    txt = txt.replace("do_loop_closures: false", f"do_loop_closures: {val}")
    Path(dst_path).write_text(txt)


# ─── RUN ONE (method, sequence, trial) ────────────────────────────────────────

def run_sequence(method: dict, seq: str, trial_idx: int) -> dict:
    """Run one (method, sequence, trial); return paths to recorded artefacts."""
    # Sweep any orphans from a previous trial/run before we spawn ours, so they
    # don't hold DDS readers on the topics we're about to publish to.
    _pkill_ros_orphans()

    bag_dir  = os.path.join(DATASET_DIR, _DS["bag_play_dir"].format(seq=seq))
    bag_file = os.path.join(DATASET_DIR, _DS["bag_pattern"].format(seq=seq))
    if not os.path.exists(bag_file):
        return {"error": f"Bag not found: {bag_file}"}

    out_dir = os.path.join(RESULTS_DIR, method['label'], seq)
    os.makedirs(out_dir, exist_ok=True)

    run_tag      = f"run{trial_idx:02d}"
    traj_dest    = os.path.join(out_dir, f"vio_trajectory_{run_tag}.tum")
    raw_traj_dest = os.path.join(out_dir, f"vio_trajectory_raw_{run_tag}.tum") if method.get('raw_odom_topic') else None
    tracker_dest = os.path.join(out_dir, f"feature_tracker_{run_tag}.csv")
    estim_dest   = os.path.join(out_dir, f"estimator_metrics_{run_tag}.csv")
    loop_dest    = os.path.join(out_dir, f"loop_fusion_metrics_{run_tag}.csv")
    loopopt_dest = os.path.join(out_dir, f"loop_optimizations_{run_tag}.csv")

    if os.path.exists(traj_dest) and os.path.getsize(traj_dest) > 0:
        print(f"    [skip] {run_tag} — trajectory already exists")
        return {
            "trial": trial_idx,
            "traj":    traj_dest,
            "traj_raw": raw_traj_dest if (raw_traj_dest and os.path.exists(raw_traj_dest)) else None,
            "tracker": tracker_dest if os.path.exists(tracker_dest) else None,
            "estim":   estim_dest   if os.path.exists(estim_dest)   else None,
            "loop":    loop_dest    if os.path.exists(loop_dest)    else None,
            "loopopt": loopopt_dest if os.path.exists(loopopt_dest) else None,
        }

    if method['runner'] == 'dl_vins':
        workspace_dir = WORKSPACE_DIR
        setup_cmd = f"source {WORKSPACE_DIR}/install/setup.bash"
        loop_arg = " use_loop_fusion:=true" if method.get('use_loop_fusion') else ""
        seq_arg = f" seq:={seq}" if _DS.get("seq_in_launch") else ""
        launch_cmd = (
            f"ros2 launch dl_vins {method['launch_file']} "
            f"extractor:={method['extractor']}{seq_arg}{loop_arg}"
        )
        node_name = "component_container"
        run_cwd  = out_dir
        log_dir  = os.path.join(out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
    elif method['runner'] == 'vins_fusion':
        workspace_dir = VINS_FUSION_DIR
        setup_cmd = f"source {VINS_FUSION_DIR}/install/setup.bash"
        cfg_path = method['config']
        # Per-sequence config override (e.g. subt_mrs has different hardware per seq).
        if cfg_path is None:
            _seq_vf = _DS.get("seq_vf_configs", {}).get(seq)
            if _seq_vf:
                cfg_path = os.path.join(VF_CONFIG_DIR, _seq_vf)
        if not cfg_path:
            return {"error": f"No VINS-Fusion config for dataset={DATASET} seq={seq}"}
        cfg_path = str(cfg_path)  # absolute path under scripts/vins_fusion_configs/
        os.makedirs("/tmp/vins_output/pose_graph", exist_ok=True)
        # vins_node is launched via the (dataset-agnostic) euroc.launch.py so its
        # odometry is remapped to /vins_estimator/odometry; loop_fusion_node
        # republishes the corrected stream on /odometry_rect.
        if method.get('use_loop_fusion'):
            log_prefix = "RCUTILS_LOGGING_BUFFERED_STREAM=0 RCUTILS_LOGGING_USE_STDOUT=1 stdbuf -oL -eL"
            launch_cmd = (
                f"({setup_cmd} && {log_prefix} ros2 launch vins euroc.launch.py config_path:={cfg_path}) & "
                f"({setup_cmd} && {log_prefix} ros2 run loop_fusion loop_fusion_node {cfg_path}) & "
                f"wait"
            )
        else:
            launch_cmd = f"ros2 launch vins euroc.launch.py config_path:={cfg_path}"
        node_name = "vins_node"
        run_cwd = workspace_dir
        log_dir = None
    elif method['runner'] == 'okvis2':
        # okvis_node_subscriber subscribes to fixed /okvis/* topics; the dataset bag
        # is remapped into them at `ros2 bag play` (bag_remap, applied below).
        # do_loop_closures (slam vs vio) is set by materialising a per-run config copy.
        # `output_csv_dir` makes the node write the keyframe trajectory as
        # okvis2-<mode>-{live,final}_trajectory.csv (rpg/TUM); we read that, not from topic.
        setup_cmd = f"source /opt/ros/humble/setup.bash && source {OKVIS2_SETUP}"
        workspace_dir = os.path.dirname(os.path.dirname(OKVIS2_SETUP))  # …/install -> ws
        okvis_src = method['okvis_config']
        # Per-sequence config override (e.g. subt_mrs has different hardware per seq).
        if okvis_src is None:
            _seq_ok = _DS.get("seq_okvis_configs", {}).get(seq)
            if _seq_ok:
                okvis_src = os.path.join(OKVIS2_CONFIG_DIR, _seq_ok)
        if not okvis_src:
            return {"error": f"No OKVIS2 config for dataset={DATASET} seq={seq}"}
        run_cfg = os.path.join(out_dir, f"okvis_config_{run_tag}.yaml")
        _write_okvis_config(str(okvis_src), run_cfg, method['okvis_do_loop'])
        launch_cmd = (
            "ros2 run okvis okvis_node_subscriber --ros-args "
            f"-p config_filename:={run_cfg} "
            f"-p output_csv_dir:={out_dir}"
        )
        node_name = "okvis_node"
        run_cwd = out_dir
        log_dir = None
    else:
        return {"error": f"Unknown runner: {method['runner']}"}

    log_env = "RCUTILS_LOGGING_BUFFERED_STREAM=0 RCUTILS_LOGGING_USE_STDOUT=1"
    if method['runner'] == 'vins_fusion' and method.get('use_loop_fusion'):
        full_launch = f"bash -c '{launch_cmd}'"
    else:
        full_launch = f"bash -c '{setup_cmd} && {log_env} stdbuf -oL -eL {launch_cmd}'"

    start_time = time.time()

    vio_stdout_path = os.path.join(out_dir, f"vio_stdout_{run_tag}.log")
    vio_stderr_path = os.path.join(out_dir, f"vio_stderr_{run_tag}.log")
    vio_stdout_f = open(vio_stdout_path, 'w')
    vio_stderr_f = open(vio_stderr_path, 'w')

    vio_proc = subprocess.Popen(
        full_launch, shell=True,
        stdout=vio_stdout_f, stderr=vio_stderr_f,
        cwd=run_cwd, start_new_session=True,
    )

    # okvis2 writes its trajectory to CSV (odom_topic is None) — no topic recorder.
    recorder_proc = None
    if method.get('odom_topic'):
        recorder_stderr_path = os.path.join(out_dir, f"recorder_stderr_{run_tag}.log")
        recorder_proc = record_trajectory(
            setup_cmd, workspace_dir, traj_dest, method['odom_topic'],
            stderr_path=recorder_stderr_path,
            msg_type=method.get('odom_msg_type', 'odometry'))

    raw_recorder_proc = None
    if raw_traj_dest and method.get('raw_odom_topic'):
        raw_recorder_stderr_path = os.path.join(out_dir, f"recorder_raw_stderr_{run_tag}.log")
        raw_recorder_proc = record_trajectory(
            setup_cmd, workspace_dir, raw_traj_dest, method['raw_odom_topic'],
            stderr_path=raw_recorder_stderr_path)

    node_pid = find_ros_node_pid(vio_proc.pid, node_name, timeout=15)
    if node_pid:
        print(f"    {method['label']} node PID: {node_pid}")
    else:
        print(f"    WARN: could not locate node '{node_name}' in process tree")

    time.sleep(4)

    if vio_proc.poll() is not None:
        kill_proc_group(recorder_proc)
        if raw_recorder_proc is not None:
            kill_proc_group(raw_recorder_proc)
        vio_stdout_f.close(); vio_stderr_f.close()
        tail = open(vio_stderr_path).read()[-500:]
        return {"error": f"VIO exited early (code={vio_proc.returncode}); stderr={tail!r}"}

    # OKVIS2 reads fixed /okvis/* topics; remap the bag's image/imu topics into them.
    remap_arg = ""
    if method.get('bag_remap'):
        remap_arg = " --remap " + " ".join(method['bag_remap'])
    gate_topic = _image_gate_topic(method)
    start_paused = " --start-paused" if gate_topic else ""
    bag_cmd = (f"bash -c '{setup_cmd} && "
               f"ros2 bag play {bag_dir} --rate {BAG_RATE} --clock 200{start_paused}{remap_arg}'")
    bag_stderr_path = os.path.join(out_dir, f"bag_stderr_{run_tag}.log")
    # ros2 bag play can refuse to exit after EOF (DDS / leftover subscribers),
    # which would otherwise block this trial forever.
    bag_timeout = bag_duration_seconds(bag_dir) / max(BAG_RATE, 1e-3) * BAG_TIMEOUT_SLOWDOWN + 60.0
    bag_stderr_f = open(bag_stderr_path, 'w')
    bag_proc = subprocess.Popen(
        bag_cmd, shell=True, cwd=workspace_dir,
        stdout=subprocess.DEVNULL, stderr=bag_stderr_f,
        start_new_session=True,
    )
    if gate_topic:
        if wait_for_pub_sub_match(setup_cmd, gate_topic, IMAGE_GATE_TIMEOUT):
            time.sleep(1.0)  # let the match settle on the player side before flow
        else:
            print(f"    WARN: {gate_topic} never matched pub+sub in "
                  f"{IMAGE_GATE_TIMEOUT:.0f}s — resuming bag anyway")
        resume_cmd = ("bash -c \"" + setup_cmd + " && ros2 service call "
                      "/rosbag2_player/resume rosbag2_interfaces/srv/Resume '{}'\"")
        try:
            resume = subprocess.run(resume_cmd, shell=True, capture_output=True,
                                    text=True, timeout=15)
            if resume.returncode != 0:
                print(f"    WARN: bag resume failed (rc={resume.returncode}) "
                      f"{resume.stderr.strip()[-200:]!r}")
        except subprocess.SubprocessError as e:
            print(f"    WARN: bag resume call errored: {e}")
    try:
        bag_rc = bag_proc.wait(timeout=bag_timeout)
    except subprocess.TimeoutExpired:
        print(f"    WARN: bag play exceeded {bag_timeout:.0f}s — force-killing")
        kill_proc_group(bag_proc, signal.SIGTERM, wait=3.0)
        if bag_proc.poll() is None:
            kill_proc_group(bag_proc, signal.SIGKILL, wait=2.0)
        bag_rc = bag_proc.returncode
    finally:
        bag_stderr_f.close()
    if bag_rc not in (0, -signal.SIGTERM, -signal.SIGKILL):
        tail = open(bag_stderr_path).read()[-300:]
        print(f"    WARN: bag play exit={bag_rc} {tail!r}")

    time.sleep(SETTLE_TIME)

    for _rec in (recorder_proc, raw_recorder_proc):
        if _rec is None:
            continue
        kill_proc_group(_rec, signal.SIGINT)
        if getattr(_rec, '_stderr_file', None):
            try:
                _rec._stderr_file.close()
            except OSError:
                pass
        if hasattr(_rec, '_script_file'):
            try:
                os.unlink(_rec._script_file)
            except OSError:
                pass
    # okvis2 writes its final trajectory CSV on SIGINT — give it time before escalating.
    if method['runner'] == 'okvis2' and vio_proc.poll() is None:
        try:
            os.killpg(os.getpgid(vio_proc.pid), signal.SIGINT)
            vio_proc.wait(timeout=30.0)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass
    kill_vio_cascade(vio_proc)
    vio_stdout_f.close(); vio_stderr_f.close()

    # okvis2: copy the optimised-trajectory CSV (rpg/TUM) the node wrote → traj_dest.
    if method['runner'] == 'okvis2':
        _okmode = method.get('okvis_mode', 'slam' if method.get('okvis_do_loop') else 'vio')
        # okvis appends a "-calib" infix when online calibration is enabled,
        # e.g. okvis2-slam-calib-final_trajectory.csv — so glob over the infix.
        for _pat in (f"okvis2-{_okmode}*-final_trajectory.csv",
                     f"okvis2-{_okmode}*-live_trajectory.csv"):
            _matches = sorted(glob.glob(os.path.join(out_dir, _pat)))
            _src = next((m for m in _matches if os.path.getsize(m) > 0), None)
            if _src:
                shutil.copy2(_src, traj_dest)
                break

    if vio_proc.returncode and vio_proc.returncode not in (-signal.SIGTERM, -signal.SIGINT):
        tail = open(vio_stderr_path).read()[-800:]
        if tail.strip():
            print(f"    VIO exit={vio_proc.returncode}; stderr tail: {tail}")

    tracker_src = estim_src = loop_src = loopopt_src = None
    if log_dir and os.path.isdir(log_dir):
        tag = method['tracker_method']
        tracker_src = find_latest_csv(log_dir, f"feature_tracker_{tag}_", start_time)
        estim_src   = find_latest_csv(log_dir, "estimator_metrics_",     start_time)
        loop_src    = find_latest_csv(log_dir, "loop_fusion_metrics_",   start_time)
        loopopt_src = find_latest_csv(log_dir, "loop_optimizations_",    start_time)

        if tracker_src:
            shutil.copy2(tracker_src, tracker_dest)
        if estim_src:
            shutil.copy2(estim_src, estim_dest)
        if loop_src:
            shutil.copy2(loop_src, loop_dest)
        if loopopt_src:
            shutil.copy2(loopopt_src, loopopt_dest)

        try:
            shutil.rmtree(log_dir)
        except OSError:
            pass

    traj_ok = os.path.exists(traj_dest) and os.path.getsize(traj_dest) > 0
    traj_raw_ok = bool(raw_traj_dest and os.path.exists(raw_traj_dest) and os.path.getsize(raw_traj_dest) > 0)
    return {
        "trial":   trial_idx,
        "traj":    traj_dest    if traj_ok                                  else None,
        "traj_raw": raw_traj_dest if traj_raw_ok                            else None,
        "tracker": tracker_dest if os.path.exists(tracker_dest)             else None,
        "estim":   estim_dest   if os.path.exists(estim_dest)               else None,
        "loop":    loop_dest    if os.path.exists(loop_dest)                else None,
        "loopopt": loopopt_dest if os.path.exists(loopopt_dest)             else None,
        "error":   None if traj_ok else "No trajectory poses recorded",
    }


def get_method(label: str) -> dict | None:
    for m in METHODS:
        if m['label'] == label:
            return m
    return None


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="DL-VINS multi-dataset eval runner — one trial per invocation.")
    p.add_argument("--method", help="Method label (see --list)")
    p.add_argument("--sequence", help="Sequence name (see --list)")
    p.add_argument("--trial", type=int, default=1, help="Trial index (default 1)")
    p.add_argument("--list", action="store_true",
                   help="Print resolved config (dataset, mode, methods, sequences, paths) and exit.")
    args = p.parse_args(argv)

    if args.list:
        print(f"DATASET          = {DATASET}")
        print(f"LAUNCH_PREFIX    = {_DS['launch_prefix']}")
        print(f"MODE             = {MODE}")
        print(f"RUN_TAG          = {RUN_TAG!r}")
        print(f"USE_LOOP_FUSION  = {USE_LOOP_FUSION}")
        print(f"WORKSPACE_DIR    = {WORKSPACE_DIR}")
        print(f"VINS_FUSION_DIR  = {VINS_FUSION_DIR}")
        print(f"DATASET_DIR      = {DATASET_DIR}")
        print(f"GT_DIR           = {GT_DIR}")
        print(f"RESULTS_DIR      = {RESULTS_DIR}")
        print(f"ROS_DOMAIN_ID    = {ROS_DOMAIN_ID}")
        print(f"SEQUENCES ({len(SEQUENCES)}): {' '.join(SEQUENCES)}")
        print(f"METHODS ({len(METHODS)}):")
        for m in METHODS:
            _cfg = m.get('config') or m.get('okvis_config')
            suffix = (m.get('extractor')
                      or ('[per-seq]' if _cfg is None and _DS.get('seq_vf_configs')
                          else os.path.basename(_cfg or '')))
            if m.get('runner') == 'okvis2':
                extras = " +loop(slam)" if m.get('okvis_do_loop') else " (vio)"
            else:
                extras = " +loop" if m.get('use_loop_fusion') else ""
            raw = f"  (raw: {m['raw_odom_topic']})" if m.get('raw_odom_topic') else ""
            print(f"  - {m['label']:34s} [{m['runner']}] {suffix}{extras} -> {m['odom_topic']}{raw}")
        return 0

    if not args.method or not args.sequence:
        p.error("--method and --sequence are required (or use --list)")

    method = get_method(args.method)
    if method is None:
        labels = ", ".join(m['label'] for m in METHODS)
        print(f"Unknown method label: {args.method!r}\nAvailable: {labels}", file=sys.stderr)
        return 2
    if args.sequence not in SEQUENCES:
        print(f"Unknown sequence: {args.sequence!r}\nAvailable: {' '.join(SEQUENCES)}", file=sys.stderr)
        return 2

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[{args.method} / {args.sequence} / run{args.trial:02d}] ROS_DOMAIN_ID={ROS_DOMAIN_ID}")
    result = run_sequence(method, args.sequence, args.trial)

    if result.get('error'):
        print(f"    ERROR: {result['error']}")
        return 1
    print(f"    traj    → {result['traj']}")
    if result.get('tracker'):
        print(f"    tracker → {result['tracker']}")
    if result.get('estim'):
        print(f"    estim   → {result['estim']}")
    if result.get('loop'):
        print(f"    loop    → {result['loop']}")
    if result.get('loopopt'):
        print(f"    loopopt → {result['loopopt']}")
    if result.get('traj_raw'):
        print(f"    traj_raw→ {result['traj_raw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
