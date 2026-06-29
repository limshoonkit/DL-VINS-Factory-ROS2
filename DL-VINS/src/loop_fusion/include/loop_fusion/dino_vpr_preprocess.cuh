#ifndef LOOP_FUSION_DINO_VPR_PREPROCESS_CUH
#define LOOP_FUSION_DINO_VPR_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

namespace uosm::loop_fusion
{
    // Fused DINOv2 preprocess in a single kernel launch:
    //   bilinear resize (aspect-preserving) into a letterboxed
    //   (target_height x target_width) canvas, gray/BGR -> RGB channel order,
    //   /255, then per-channel ImageNet mean/std normalization, written as
    //   NCHW float32 to `output` (3 * target_height * target_width).
    //
    // The padded border is filled with the normalized zero value
    // ((0 - mean) / std), matching
    // AnyLoc/vpr/dino_vpr_export.py::_letterbox_normalized exactly so the
    // runtime tokens match the distribution the VLAD vocabulary was fit on.
    void launchDinoPreprocess(
        const cv::cuda::GpuMat &img,
        float *output,
        int target_height,
        int target_width,
        int new_w,
        int new_h,
        int pad_x,
        int pad_y,
        cudaStream_t stream);
} // namespace uosm::loop_fusion

#endif // LOOP_FUSION_DINO_VPR_PREPROCESS_CUH
