#!/usr/bin/env bash
# Run the LET-VINS (VINS-Mono + LET-NET) baseline on a SubT-MRS handheld sequence in the
# letvins:noetic container, then score it against GT with evo pipeline.
#
# Usage: scripts/letnet_baseline.sh <low_light1|low_light2|over_exposure|flash_light|smoke_handheld> [playback_rate]
set -euo pipefail

SEQ="${1:?usage: letnet_baseline.sh <low_light1|low_light2|over_exposure|flash_light|smoke_handheld> [rate]}"
RATE="${2:-1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LETROOT="$REPO_ROOT/baseline_reference/LET-NET"
MODEL="$LETROOT/model"
DATASET_ROOT="$REPO_ROOT/dataset/6_SubT-MRS"
case "$SEQ" in
  low_light1)    BAGDIR="$DATASET_ROOT/SubT_MRS_Laurel_Caverns_Handheld1/Handheld1_Rosbag" ;;
  low_light2)    BAGDIR="$DATASET_ROOT/SubT_MRS_Laurel_Caverns_Handheld2/Handheld2_Rosbag" ;;
  flash_light)   BAGDIR="$DATASET_ROOT/SubT_MRS_Laurel_Caverns_RC7_FlashLight/RC7_FlashLight_Rosbag" ;;
  over_exposure) BAGDIR="$DATASET_ROOT/SubT_MRS_Laurel_Caverns_RC7_OverExposure/RC7_OverExposure_Rosbag" ;;
  smoke_handheld) BAGDIR="$DATASET_ROOT/SubT_MRS_Laurel_Caverns_Handheld_Smoke/HandheldSmoke_Rosbag" ;;
  *) echo "unknown seq: $SEQ"; exit 1 ;;
esac

OUT="$REPO_ROOT/tmp/letnet_baseline/$SEQ"
mkdir -p "$OUT"
echo ">> $SEQ : bags=$BAGDIR  rate=$RATE  out=$OUT"

docker run --rm \
  -v "$MODEL:/root/let_net_model:ro" \
  -v "$BAGDIR:/bags:ro" \
  -v "$OUT:/root/output" \
  -e LET_NET_MODEL_DIR=/root/let_net_model \
  -e LET_VIZ_DIR=/root/output/viz_frames \
  -e LET_VIZ_STRIDE=10 \
  letvins:noetic \
  bash -lc '
    set -e
    source /root/catkin_ws/devel/setup.bash
    mkdir -p /root/output/pose_graph /root/output/viz_frames
    rm -f /root/output/vins_result_no_loop.csv /root/output/vins_result_loop.csv /root/output/traj.txt
    rm -f /root/output/viz_frames/*.jpg 2>/dev/null || true
    roscore >/root/output/roscore.log 2>&1 & sleep 4
    roslaunch vins_estimator subt.launch >/root/output/vins.log 2>&1 & sleep 14
    echo ">> playing bags..."
    rosbag play -r '"$RATE"' --quiet /bags/raw_data_core_*_*.bag
    echo ">> playback done, flushing..."; sleep 10
    pkill -f roslaunch 2>/dev/null || true; pkill -f vins_estimator 2>/dev/null || true
    pkill -f pose_graph 2>/dev/null || true; pkill -f feature_tracker 2>/dev/null || true
    pkill rosmaster 2>/dev/null || true; sleep 2
  '
