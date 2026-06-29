#!/usr/bin/env bash
# Multi-dataset VIO/SLAM evaluation entrypoint.
#
# Default method set is the classic (EVAL_METHOD_SET=classic):
#   OKVIS2  vs  VINS-Fusion,  loop closure ON,
#   recording base (no-PGO) and corrected trajectories for each.
#
# Override defaults via env vars (also picked up by eval_runner.py):
#   EVAL_DATASET=euroc|ntu_viral|botanic_garden|subt_mrs  (default: euroc; ignored if EVAL_QUICK=1)
#   EVAL_DATASETS="euroc ntu_viral ..."  (datasets to sweep; default: EVAL_DATASET)
#   EVAL_QUICK=1                     (sweep all 4 datasets, 2 sequences each)
#   EVAL_MODE=mono|stereo            (default: mono)
#   EVAL_METHOD_SET=classic|dl       (default: classic)
#   EVAL_USE_LOOP_FUSION=true|false  (default: true — base+corrected per method)
#   EVAL_RUN_TAG=<suffix>            (default: "")
#   EVAL_SEQUENCES="MH_01_easy ..."  (default: full set, or quick set if EVAL_QUICK=1)
#   EVAL_METHODS="vins_fusion_mono_loop ..."  (default: all from --list)
#   EVAL_TRIALS=<n>                  (default: 1)
#   EVAL_SKIP_CLOCK_LOCK=1           (skip nvpmodel/jetson_clocks — for dev)
#   EVAL_SKIP_TEGRASTATS=1           (skip tegrastats logging)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/eval_runner.py"

# CycloneDDS socket-buffer tuning (64 MiB receive).
# export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://$SCRIPT_DIR/cyclonedds.xml}"

# Loop closure on by default — the comparison records base + corrected per method.
export EVAL_USE_LOOP_FUSION="${EVAL_USE_LOOP_FUSION:-true}"
export EVAL_METHOD_SET="${EVAL_METHOD_SET:-classic}"

# Quick-test sequence subset (2 per dataset). Overridden by EVAL_SEQUENCES.
quick_sequences() {
    case "$1" in
        euroc)          echo "MH_03_medium V2_02_medium" ;;
        ntu_viral)      echo "eee_01 nya_02" ;;
        botanic_garden) echo "1005_00 1006_01" ;;
        botanic_garden_rgb) echo "1006_01 1018_13" ;;
        subt_mrs)       echo "flash_light over_exposure" ;;
        *)              echo "" ;;
    esac
}

# Platform detection: Jetson (aarch64 + nvpmodel) vs x86_64 dev box.
UNAME_A="$(uname -a)"
ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" ]] && command -v nvpmodel >/dev/null 2>&1; then
    IS_JETSON=1
else
    IS_JETSON=0
fi
echo "[eval_run] uname: $UNAME_A"
echo "[eval_run] IS_JETSON=$IS_JETSON (arch=$ARCH)"

if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_cyclonedds_cpp" ]]; then
    echo "ERROR: RMW_IMPLEMENTATION is '${RMW_IMPLEMENTATION:-unset}', expected 'rmw_cyclonedds_cpp'." >&2
    echo "Set 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' in ~/.bashrc and open a new shell." >&2
    exit 1
fi

if [[ "$IS_JETSON" == "1" ]]; then
    if [[ "${EVAL_SKIP_CLOCK_LOCK:-0}" != "1" ]]; then
        echo "[eval_run] locking Jetson clocks: nvpmodel -m 0 && jetson_clocks"
        sudo nvpmodel -m 0
        sudo jetson_clocks
    else
        echo "[eval_run] EVAL_SKIP_CLOCK_LOCK=1 — leaving DVFS state untouched"
    fi
else
    echo "[eval_run] non-Jetson host — skipping nvpmodel/jetson_clocks"
fi

# Run the full (method × sequence × trial) grid for one dataset.
run_one_dataset() {
    local DS="$1"
    export EVAL_DATASET="$DS"

    # Resolve config via eval_runner --list (single source of truth for this dataset).
    local RESULTS_DIR ALL_SEQUENCES ALL_METHODS SEQUENCES METHODS TRIALS
    RESULTS_DIR=$(python3 "$RUNNER" --list | awk -F' = ' '/^RESULTS_DIR/{print $2}')
    ALL_SEQUENCES=$(python3 "$RUNNER" --list | awk '/^SEQUENCES /{sub(/^[^:]+: /,""); print}')
    ALL_METHODS=$(python3 "$RUNNER" --list | awk '/^  - /{print $2}' | tr '\n' ' ')

    if [[ -n "${EVAL_SEQUENCES:-}" ]]; then
        SEQUENCES="$EVAL_SEQUENCES"
    elif [[ "${EVAL_QUICK:-0}" == "1" ]]; then
        SEQUENCES="$(quick_sequences "$DS")"
    else
        SEQUENCES="$ALL_SEQUENCES"
    fi
    METHODS="${EVAL_METHODS:-$ALL_METHODS}"
    TRIALS="${EVAL_TRIALS:-1}"

    echo ""
    echo "[eval_run] ===== DATASET=$DS  MODE=${EVAL_MODE:-mono}  LOOP=${EVAL_USE_LOOP_FUSION} ====="
    echo "[eval_run] RESULTS_DIR=$RESULTS_DIR"
    echo "[eval_run] METHODS    = $METHODS"
    echo "[eval_run] SEQUENCES  = $SEQUENCES"
    echo "[eval_run] TRIALS     = $TRIALS"

    mkdir -p "$RESULTS_DIR"

    # GT is pre-baked: GT_DIR holds <seq>_gt.tum (extracted offline). No conversion here.

    for METHOD in $METHODS; do
        for SEQ in $SEQUENCES; do
            for ((T=1; T<=TRIALS; T++)); do
                OUT="$RESULTS_DIR/$METHOD/$SEQ"
                mkdir -p "$OUT"

                TAG=$(printf 'run%02d' "$T")
                ENV_FILE="$OUT/trial_env_${TAG}.txt"
                TG_LOG="$OUT/tegrastats_${TAG}.log"

                {
                    echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
                    echo "EVAL_DATASET=$DS"
                    echo "EVAL_MODE=${EVAL_MODE:-mono}"
                    echo "EVAL_METHOD_SET=${EVAL_METHOD_SET}"
                    echo "EVAL_RUN_TAG=${EVAL_RUN_TAG:-}"
                    echo "EVAL_USE_LOOP_FUSION=${EVAL_USE_LOOP_FUSION}"
                    echo "arch=$ARCH"
                    echo "is_jetson=$IS_JETSON"
                    if [[ "$IS_JETSON" == "1" ]]; then
                        echo "nvpmodel: $(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
                    fi
                    printf 'scaling_cur_freq: '
                    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | tr '\n' ' '
                    echo
                    echo "started_at=$(date -Iseconds)"
                } > "$ENV_FILE"

                TG_PID=""
                if [[ "$IS_JETSON" == "1" ]] \
                    && [[ "${EVAL_SKIP_TEGRASTATS:-0}" != "1" ]] \
                    && command -v tegrastats >/dev/null 2>&1; then
                    tegrastats --interval 200 --logfile "$TG_LOG" &
                    TG_PID=$!
                fi

                set +e
                python3 "$RUNNER" --method "$METHOD" --sequence "$SEQ" --trial "$T"
                RC=$?
                set -e

                if [[ -n "$TG_PID" ]]; then
                    kill "$TG_PID" 2>/dev/null || true
                    wait "$TG_PID" 2>/dev/null || true
                fi

                echo "ended_at=$(date -Iseconds)" >> "$ENV_FILE"
                echo "runner_exit_code=$RC" >> "$ENV_FILE"
                if [[ "$RC" -ne 0 ]]; then
                    echo "[eval_run] WARN $METHOD/$SEQ/$TAG exit=$RC — see $OUT/vio_stderr_${TAG}.log" >&2
                fi
            done
        done
    done

    echo "[eval_run] dataset $DS done. Results in $RESULTS_DIR"
}

# Which datasets to sweep.
if [[ -n "${EVAL_DATASETS:-}" ]]; then
    DATASETS="$EVAL_DATASETS"
elif [[ "${EVAL_QUICK:-0}" == "1" ]]; then
    DATASETS="euroc ntu_viral botanic_garden subt_mrs"
else
    DATASETS="${EVAL_DATASET:-euroc}"
fi
echo "[eval_run] DATASETS = $DATASETS  (EVAL_QUICK=${EVAL_QUICK:-0})"

for DS in $DATASETS; do
    run_one_dataset "$DS"
done

echo "[eval_run] all done."
