#!/usr/bin/env bash
#
# Build and install DL-VINS/Third_Party dependencies for Jetson (JetPack 6.2 / aarch64).
#
# Builds:
#   - OpenCV 4.13.0 (+ opencv_contrib) with CUDA + cuDNN  ->  DL-VINS/Third_Party/install/opencv
#   - Ceres Solver                                        ->  DL-VINS/Third_Party/install/ceres
#
# TensorRT: uses the system packages from JetPack (libnvinfer-dev, libnvinfer-plugin-dev,
# libnvonnxparsers-dev). No build needed.
#
# Usage:
#   scripts/install_third_party.sh [--opencv-only|--ceres-only] [--no-deps] [--clean] [-j N]

set -euo pipefail

# --- paths ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TP_DIR="${REPO_ROOT}/DL-VINS/Third_Party"
INSTALL_ROOT="${TP_DIR}/install"
OPENCV_SRC="${TP_DIR}/opencv"
OPENCV_CONTRIB="${TP_DIR}/opencv_contrib"
CERES_SRC="${TP_DIR}/ceres"
OPENCV_PREFIX="${INSTALL_ROOT}/opencv"
CERES_PREFIX="${INSTALL_ROOT}/ceres"

# --- defaults ------------------------------------------------------------
DO_OPENCV=1
DO_CERES=1
DO_DEPS=1
CLEAN=0
JOBS="${MAKE_JOBS:-4}"
CUDA_ARCH_BIN="${CUDA_ARCH_BIN:-8.7}"   # Jetson Orin = SM 8.7

# --- args ----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --opencv-only) DO_CERES=0 ;;
        --ceres-only)  DO_OPENCV=0 ;;
        --no-deps)     DO_DEPS=0 ;;
        --clean)       CLEAN=1 ;;
        -j)            JOBS="$2"; shift ;;
        -h|--help)
            sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n 20
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# --- sanity checks -------------------------------------------------------
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Warning: this script targets Jetson aarch64; running on $(uname -m)." >&2
fi
if ! command -v nvcc >/dev/null; then
    echo "Error: nvcc not found. Source CUDA env (e.g. add /usr/local/cuda/bin to PATH)." >&2
    exit 1
fi
if [[ ${DO_OPENCV} -eq 1 && ! -f "${OPENCV_SRC}/CMakeLists.txt" ]]; then
    echo "Error: OpenCV source not found at ${OPENCV_SRC}. Run: git submodule update --init --recursive" >&2
    exit 1
fi
if [[ ${DO_OPENCV} -eq 1 && ! -d "${OPENCV_CONTRIB}/modules" ]]; then
    echo "Error: opencv_contrib not found at ${OPENCV_CONTRIB}. Run: git submodule update --init --recursive" >&2
    exit 1
fi
if [[ ${DO_CERES} -eq 1 && ! -f "${CERES_SRC}/CMakeLists.txt" ]]; then
    echo "Error: Ceres source not found at ${CERES_SRC}. Run: git submodule update --init --recursive" >&2
    exit 1
fi

mkdir -p "${INSTALL_ROOT}"

# --- apt deps ------------------------------------------------------------
install_apt_deps() {
    echo "==> Installing build dependencies via apt..."
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config ninja-build \
        libgtk-3-dev \
        libavcodec-dev libavformat-dev libswscale-dev libavutil-dev libswresample-dev \
        libv4l-dev libxvidcore-dev libx264-dev \
        libjpeg-dev libpng-dev libtiff-dev libopenexr-dev libwebp-dev \
        libtbb-dev libeigen3-dev \
        libgoogle-glog-dev libgflags-dev libsuitesparse-dev libatlas-base-dev liblapack-dev \
        libprotobuf-dev protobuf-compiler \
        python3-dev python3-numpy
}

# --- OpenCV --------------------------------------------------------------
build_opencv() {
    local build_dir="${OPENCV_SRC}/build"
    if [[ ${CLEAN} -eq 1 ]]; then rm -rf "${build_dir}" "${OPENCV_PREFIX}"; fi
    mkdir -p "${build_dir}"

    echo "==> Configuring OpenCV ($(git -C "${OPENCV_SRC}" describe --tags 2>/dev/null || echo unknown))"
    cmake -S "${OPENCV_SRC}" -B "${build_dir}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${OPENCV_PREFIX}" \
        -DOPENCV_EXTRA_MODULES_PATH="${OPENCV_CONTRIB}/modules" \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DOPENCV_DNN_CUDA=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_CUFFT=ON \
        -DENABLE_FAST_MATH=ON \
        -DCUDA_FAST_MATH=ON \
        -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
        -DCUDA_ARCH_PTX="" \
        -DWITH_GTK=ON \
        -DWITH_TBB=ON \
        -DWITH_V4L=ON \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=ON \
        -DWITH_OPENGL=ON \
        -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,calib3d,features2d,flann,video,objdetect,photo,stitching,dnn,cudaarithm,cudaimgproc,cudawarping,cudafilters,cudafeatures2d,cudaoptflow,cudacodec,cudev,xfeatures2d,ximgproc,aruco,optflow,tracking

    echo "==> Building OpenCV (-j${JOBS})..."
    cmake --build "${build_dir}" -j "${JOBS}"
    echo "==> Installing OpenCV to ${OPENCV_PREFIX}"
    cmake --install "${build_dir}"
}

# --- Ceres ---------------------------------------------------------------
build_ceres() {
    local build_dir="${CERES_SRC}/build"
    if [[ ${CLEAN} -eq 1 ]]; then rm -rf "${build_dir}" "${CERES_PREFIX}"; fi
    mkdir -p "${build_dir}"

    echo "==> Configuring Ceres ($(git -C "${CERES_SRC}" describe --tags 2>/dev/null || echo unknown))"
    cmake -S "${CERES_SRC}" -B "${build_dir}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${CERES_PREFIX}" \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DUSE_CUDA=ON \
        -DEIGENSPARSE=ON \
        -DSUITESPARSE=ON \
        -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}"

    echo "==> Building Ceres (-j${JOBS})..."
    cmake --build "${build_dir}" -j "${JOBS}"
    echo "==> Installing Ceres to ${CERES_PREFIX}"
    cmake --install "${build_dir}"
}

# --- run -----------------------------------------------------------------
[[ ${DO_DEPS}   -eq 1 ]] && install_apt_deps
[[ ${DO_OPENCV} -eq 1 ]] && build_opencv
[[ ${DO_CERES}  -eq 1 ]] && build_ceres

echo
echo "Done. Installed under: ${INSTALL_ROOT}"
echo "  OpenCV:  ${OPENCV_PREFIX}"
echo "  Ceres:   ${CERES_PREFIX}"
echo "TensorRT is provided by JetPack (system /usr packages); no build needed."
