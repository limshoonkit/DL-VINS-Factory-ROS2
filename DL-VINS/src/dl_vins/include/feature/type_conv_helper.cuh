#ifndef TYPE_CONV_HELPER_CUH
#define TYPE_CONV_HELPER_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <opencv2/core/cuda.hpp>

// Transform keypoints from model output to original image coordinates
// Handles letterboxing: scale + offset for both axes
void launchTransformKeypointsSingle(
    void *d_keypoints,
    void *d_num_keypoints,
    size_t num_kpts_element_size,
    float scale,    // Uniform scale (original_size / resized_size)
    float x_offset, // X padding offset in model input pixels
    float y_offset, // Y padding offset in model input pixels
    int max_keypoints,
    int target_width,  // Model input width
    int target_height, // Model input height
    bool is_aliked,
    cudaStream_t stream);

// bilinear resize + (BGR→GRAY) + ÷255 + NCHW write in one kernel.
void launchFusedPreprocess_Gray_Single(
    const cv::cuda::GpuMat &img,
    float *output,
    int target_height,
    int target_width,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    cudaStream_t stream);

void launchFusedPreprocess_RGB_Single(
    const cv::cuda::GpuMat &img,
    float *output,
    int target_height,
    int target_width,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    cudaStream_t stream);

void launchFusedPreprocess_Gray_Stereo(
    const cv::cuda::GpuMat &img_left,
    const cv::cuda::GpuMat &img_right,
    float *output,
    int target_height,
    int target_width,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    cudaStream_t stream);

void launchFusedPreprocess_RGB_Stereo(
    const cv::cuda::GpuMat &img_left,
    const cv::cuda::GpuMat &img_right,
    float *output,
    int target_height,
    int target_width,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    cudaStream_t stream);

// Transform keypoints for stereo batch (both left and right images)
void launchTransformKeypointsStereo(
    void *d_keypoints,     // Batched keypoints: (2, max_keypoints, 2)
    void *d_num_keypoints, // Batched counts: (2,)
    size_t num_kpts_element_size,
    float scale,
    float x_offset,
    float y_offset,
    int max_keypoints,
    int target_width,
    int target_height,
    bool is_aliked,
    cudaStream_t stream);

// Normalize pixel-space keypoints into model-space [-1, 1] coordinates.
void launchNormalizeKeypointsToModelSpace(
    const float *d_kps_px,
    float *d_kps_model,
    int N,
    float fwd_scale,
    float x_offset,
    float y_offset,
    int model_w,
    int model_h,
    cudaStream_t stream);

// Normalize pixel-space keypoints to LightGlue's image-space convention
// (px - center) / (0.5 * max(w,h)). Writes a fixed-size (max_kp, kpt_dim) buffer:
// indices >= n_use are zero-filled to match the engine's static input shape.
// When kpt_dim == 4 (SIFT), input layout is (x_px, y_px, scale, ori_rad);
void launchNormalizeKeypointsLightGlue(
    const float *d_kps_px, // (n_use, kpt_dim) -- read range
    float *d_kps_norm,     // (max_kp, kpt_dim) -- write range, padded with 0
    int n_use,
    int max_kp,
    int kpt_dim,           // 2 (default) or 4 (SIFT scale+ori)
    float cx,
    float cy,
    float inv_scale, // 1.0f / (0.5 * max(image_w, image_h))
    cudaStream_t stream);

// ALIKED dense-pipeline postproc: replaces the in-engine DKD with a custom kernel.
size_t alikedDkdPostprocWorkspaceBytes(int B, int H, int W, int max_K);

void launchAlikedDkdPostproc(
    const float *d_score_map, // (B, 1, H, W) FP32 from dense engine output
    int B, int H, int W,
    int max_K,                  // 256 typically
    int nms_radius,             // 2 (matches DKD radius)
    float scores_th,            // 0.0f for top-K mode
    float subpixel_temperature, // 0.1f
    float scale,                // letterbox inverse: original_px = (model_px - offset) * scale
    float x_offset,
    float y_offset,
    float *d_kpts_px, // (B, max_K, 2) — output, pixel coords
    float *d_kpts_n,  // (B, max_K, 2) — output, normalised [-1, 1]
    float *d_scores,  // (B, max_K)    — output
    int *d_num_kpts,  // (B,)          — output
    void *d_workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

// Gather kept rows from (src_kps : N_src x kpt_stride) and (src_desc : N_src x D)
// into dense (n_kept x kpt_stride) and (n_kept x D) destination buffers.
void launchGatherKeypointsAndDescriptors(
    const float *d_kps_src,    // (N_src, kpt_stride)
    const float *d_desc_src,   // (N_src, D)
    const int *d_kept_indices, // (n_kept,)
    int n_kept,
    int D,
    int kpt_stride,            // 2 or 4
    float *d_kps_dst,  // (n_kept, kpt_stride)
    float *d_desc_dst, // (n_kept, D)
    cudaStream_t stream);

#endif // TYPE_CONV_HELPER_CUH