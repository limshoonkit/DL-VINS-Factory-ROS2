#include "../include/feature/type_conv_helper.cuh"
#include <cub/cub.cuh>
#include <cstdint>

// Single image keypoint transformation with letterbox handling (batch=1)
__global__ void transform_keypoints_single_kernel(
    float *keypoints,
    void *num_keypoints_raw,
    const size_t num_kpts_element_size,
    const float scale,
    const float x_offset,
    const float y_offset,
    const int max_keypoints,
    const int target_width,
    const int target_height,
    const bool is_aliked)
{
    const int kpt_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (kpt_idx >= max_keypoints)
        return;

    int num_kpts = max_keypoints;
    if (num_keypoints_raw != nullptr)
    {
        if (num_kpts_element_size == sizeof(int32_t))
            num_kpts = reinterpret_cast<int32_t *>(num_keypoints_raw)[0];
        else
            num_kpts = static_cast<int>(reinterpret_cast<int64_t *>(num_keypoints_raw)[0]);
    }

    if (kpt_idx >= num_kpts)
        return;

    float *kpt_ptr = &keypoints[kpt_idx * 2];

    float x = kpt_ptr[0];
    float y = kpt_ptr[1];

    if (is_aliked)
    {
        x = (x + 1.0f) * 0.5f * (target_width - 1);
        y = (y + 1.0f) * 0.5f * (target_height - 1);
    }

    x = (x - x_offset) * scale;
    y = (y - y_offset) * scale;

    kpt_ptr[0] = x;
    kpt_ptr[1] = y;
}

void launchTransformKeypointsSingle(
    void *d_keypoints,
    void *d_num_keypoints,
    size_t num_kpts_element_size,
    float scale,
    float x_offset,
    float y_offset,
    int max_keypoints,
    int target_width,
    int target_height,
    bool is_aliked,
    cudaStream_t stream)
{
    dim3 blockSize(min(256, max_keypoints));
    dim3 gridSize((max_keypoints + blockSize.x - 1) / blockSize.x);

    transform_keypoints_single_kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<float *>(d_keypoints),
        d_num_keypoints,
        num_kpts_element_size,
        scale,
        x_offset,
        y_offset,
        max_keypoints,
        target_width,
        target_height,
        is_aliked);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchTransformKeypointsSingle: %s\n", cudaGetErrorString(err));
    }
}

__global__ void transform_keypoints_stereo_kernel(
    float *keypoints,        // (2, max_keypoints, 2)
    void *num_keypoints_raw, // (2,)
    const size_t num_kpts_element_size,
    const float scale,
    const float x_offset,
    const float y_offset,
    const int max_keypoints,
    const int target_width,
    const int target_height,
    const bool is_aliked)
{
    // Thread handles one keypoint from one batch
    const int batch_idx = blockIdx.y; // 0 = left, 1 = right
    const int kpt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (kpt_idx >= max_keypoints || batch_idx >= 2)
        return;

    // Get number of keypoints for this batch
    int num_kpts = max_keypoints;
    if (num_keypoints_raw != nullptr)
    {
        if (num_kpts_element_size == sizeof(int32_t))
            num_kpts = reinterpret_cast<int32_t *>(num_keypoints_raw)[batch_idx];
        else
            num_kpts = static_cast<int>(reinterpret_cast<int64_t *>(num_keypoints_raw)[batch_idx]);
    }

    if (kpt_idx >= num_kpts)
        return;

    // Compute offset into keypoints array: batch_idx * max_keypoints * 2 + kpt_idx * 2
    float *kpt_ptr = &keypoints[batch_idx * max_keypoints * 2 + kpt_idx * 2];

    float x = kpt_ptr[0];
    float y = kpt_ptr[1];

    if (is_aliked)
    {
        x = (x + 1.0f) * 0.5f * (target_width - 1);
        y = (y + 1.0f) * 0.5f * (target_height - 1);
    }

    x = (x - x_offset) * scale;
    y = (y - y_offset) * scale;

    kpt_ptr[0] = x;
    kpt_ptr[1] = y;
}

void launchTransformKeypointsStereo(
    void *d_keypoints,
    void *d_num_keypoints,
    size_t num_kpts_element_size,
    float scale,
    float x_offset,
    float y_offset,
    int max_keypoints,
    int target_width,
    int target_height,
    bool is_aliked,
    cudaStream_t stream)
{
    dim3 blockSize(min(256, max_keypoints));
    dim3 gridSize((max_keypoints + blockSize.x - 1) / blockSize.x, 2); // 2 batches

    transform_keypoints_stereo_kernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<float *>(d_keypoints),
        d_num_keypoints,
        num_kpts_element_size,
        scale,
        x_offset,
        y_offset,
        max_keypoints,
        target_width,
        target_height,
        is_aliked);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchTransformKeypointsStereo: %s\n", cudaGetErrorString(err));
    }
}

// Bilinear sampler. Returns three components in [0, 255] (broadcast for src_channels==1).
// For src_channels==3, components are (B, G, R) — i.e. raw OpenCV byte order.
// Uses OpenCV INTER_LINEAR pixel-center sampling: sx = (dst_x + 0.5) * fx - 0.5.
__device__ __forceinline__ void fused_bilinear_sample(
    const uint8_t *src, int src_step, int src_channels,
    int src_h, int src_w,
    float sx, float sy,
    float &c0, float &c1, float &c2)
{
    sx = fmaxf(0.0f, fminf(sx, static_cast<float>(src_w - 1)));
    sy = fmaxf(0.0f, fminf(sy, static_cast<float>(src_h - 1)));

    const int x0 = static_cast<int>(sx);
    const int y0 = static_cast<int>(sy);
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);

    const float fx = sx - x0;
    const float fy = sy - y0;
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    if (src_channels == 1)
    {
        const float v00 = src[y0 * src_step + x0];
        const float v01 = src[y0 * src_step + x1];
        const float v10 = src[y1 * src_step + x0];
        const float v11 = src[y1 * src_step + x1];
        c0 = c1 = c2 = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
    }
    else
    {
        const uint8_t *p00 = src + y0 * src_step + x0 * 3;
        const uint8_t *p01 = src + y0 * src_step + x1 * 3;
        const uint8_t *p10 = src + y1 * src_step + x0 * 3;
        const uint8_t *p11 = src + y1 * src_step + x1 * 3;

        c0 = p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11; // B
        c1 = p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11; // G
        c2 = p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11; // R
    }
}

__global__ void fused_preprocess_gray_single_kernel(
    const uint8_t *src, int src_step, int src_channels,
    int src_h, int src_w,
    float *dst, int dst_w,
    int pad_x, int pad_y,
    int new_w, int new_h,
    float inv_scale_x, float inv_scale_y)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= new_w || ty >= new_h)
        return;

    const float sx = (tx + 0.5f) * inv_scale_x - 0.5f;
    const float sy = (ty + 0.5f) * inv_scale_y - 0.5f;

    float c0, c1, c2;
    fused_bilinear_sample(src, src_step, src_channels, src_h, src_w, sx, sy, c0, c1, c2);

    const float gray = (src_channels == 1) ? c0
                                           : (0.114f * c0 + 0.587f * c1 + 0.299f * c2);
    dst[(pad_y + ty) * dst_w + (pad_x + tx)] = gray * (1.0f / 255.0f);
}

__global__ void fused_preprocess_rgb_single_kernel(
    const uint8_t *src, int src_step, int src_channels,
    int src_h, int src_w,
    float *dst, int dst_h, int dst_w,
    int pad_x, int pad_y,
    int new_w, int new_h,
    float inv_scale_x, float inv_scale_y)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= new_w || ty >= new_h)
        return;

    const float sx = (tx + 0.5f) * inv_scale_x - 0.5f;
    const float sy = (ty + 0.5f) * inv_scale_y - 0.5f;

    float c0, c1, c2;
    fused_bilinear_sample(src, src_step, src_channels, src_h, src_w, sx, sy, c0, c1, c2);

    // Source is BGR (3ch) or gray-broadcast (1ch). Output NCHW channel order is RGB.
    const float r = (src_channels == 1) ? c0 : c2;
    const float g = (src_channels == 1) ? c0 : c1;
    const float b = c0;

    const int channel_size = dst_h * dst_w;
    const int pixel_idx = (pad_y + ty) * dst_w + (pad_x + tx);
    const float inv_255 = 1.0f / 255.0f;
    dst[pixel_idx] = r * inv_255;
    dst[channel_size + pixel_idx] = g * inv_255;
    dst[2 * channel_size + pixel_idx] = b * inv_255;
}

__global__ void fused_preprocess_gray_stereo_kernel(
    const uint8_t *src_l, int src_step_l, int src_channels_l,
    const uint8_t *src_r, int src_step_r, int src_channels_r,
    int src_h, int src_w,
    float *dst, int dst_h, int dst_w,
    int pad_x, int pad_y,
    int new_w, int new_h,
    float inv_scale_x, float inv_scale_y)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= new_w || ty >= new_h)
        return;

    const float sx = (tx + 0.5f) * inv_scale_x - 0.5f;
    const float sy = (ty + 0.5f) * inv_scale_y - 0.5f;

    const int pixel_idx = (pad_y + ty) * dst_w + (pad_x + tx);
    const int image_size = dst_h * dst_w;
    const float inv_255 = 1.0f / 255.0f;

    float c0, c1, c2;
    fused_bilinear_sample(src_l, src_step_l, src_channels_l, src_h, src_w, sx, sy, c0, c1, c2);
    const float gray_l = (src_channels_l == 1) ? c0
                                               : (0.114f * c0 + 0.587f * c1 + 0.299f * c2);
    dst[pixel_idx] = gray_l * inv_255;

    fused_bilinear_sample(src_r, src_step_r, src_channels_r, src_h, src_w, sx, sy, c0, c1, c2);
    const float gray_r = (src_channels_r == 1) ? c0
                                               : (0.114f * c0 + 0.587f * c1 + 0.299f * c2);
    dst[image_size + pixel_idx] = gray_r * inv_255;
}

__global__ void fused_preprocess_rgb_stereo_kernel(
    const uint8_t *src_l, int src_step_l, int src_channels_l,
    const uint8_t *src_r, int src_step_r, int src_channels_r,
    int src_h, int src_w,
    float *dst, int dst_h, int dst_w,
    int pad_x, int pad_y,
    int new_w, int new_h,
    float inv_scale_x, float inv_scale_y)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= new_w || ty >= new_h)
        return;

    const float sx = (tx + 0.5f) * inv_scale_x - 0.5f;
    const float sy = (ty + 0.5f) * inv_scale_y - 0.5f;

    const int pixel_idx = (pad_y + ty) * dst_w + (pad_x + tx);
    const int channel_size = dst_h * dst_w;
    const int image_size = 3 * channel_size;
    const float inv_255 = 1.0f / 255.0f;

    float c0, c1, c2;
    fused_bilinear_sample(src_l, src_step_l, src_channels_l, src_h, src_w, sx, sy, c0, c1, c2);
    {
        const float r = (src_channels_l == 1) ? c0 : c2;
        const float g = (src_channels_l == 1) ? c0 : c1;
        const float b = c0;
        dst[pixel_idx] = r * inv_255;
        dst[channel_size + pixel_idx] = g * inv_255;
        dst[2 * channel_size + pixel_idx] = b * inv_255;
    }

    fused_bilinear_sample(src_r, src_step_r, src_channels_r, src_h, src_w, sx, sy, c0, c1, c2);
    {
        const float r = (src_channels_r == 1) ? c0 : c2;
        const float g = (src_channels_r == 1) ? c0 : c1;
        const float b = c0;
        dst[image_size + pixel_idx] = r * inv_255;
        dst[image_size + channel_size + pixel_idx] = g * inv_255;
        dst[image_size + 2 * channel_size + pixel_idx] = b * inv_255;
    }
}

static inline float fused_inv_scale(int src_dim, int new_dim)
{
    return static_cast<float>(src_dim) / static_cast<float>(new_dim);
}

void launchFusedPreprocess_Gray_Single(
    const cv::cuda::GpuMat &img,
    float *output,
    int target_height,
    int target_width,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    cudaStream_t stream)
{
    (void)target_height; // dst_w is the only model dim needed for 1ch NCHW indexing
    dim3 blockSize(16, 16);
    dim3 gridSize((new_w + blockSize.x - 1) / blockSize.x,
                  (new_h + blockSize.y - 1) / blockSize.y);

    fused_preprocess_gray_single_kernel<<<gridSize, blockSize, 0, stream>>>(
        img.ptr<uint8_t>(),
        static_cast<int>(img.step),
        img.channels(),
        img.rows, img.cols,
        output, target_width,
        pad_x, pad_y,
        new_w, new_h,
        fused_inv_scale(img.cols, new_w),
        fused_inv_scale(img.rows, new_h));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchFusedPreprocess_Gray_Single: %s\n", cudaGetErrorString(err));
    }
}

void launchFusedPreprocess_RGB_Single(
    const cv::cuda::GpuMat &img,
    float *output,
    int target_height,
    int target_width,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((new_w + blockSize.x - 1) / blockSize.x,
                  (new_h + blockSize.y - 1) / blockSize.y);

    fused_preprocess_rgb_single_kernel<<<gridSize, blockSize, 0, stream>>>(
        img.ptr<uint8_t>(),
        static_cast<int>(img.step),
        img.channels(),
        img.rows, img.cols,
        output, target_height, target_width,
        pad_x, pad_y,
        new_w, new_h,
        fused_inv_scale(img.cols, new_w),
        fused_inv_scale(img.rows, new_h));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchFusedPreprocess_RGB_Single: %s\n", cudaGetErrorString(err));
    }
}

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
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((new_w + blockSize.x - 1) / blockSize.x,
                  (new_h + blockSize.y - 1) / blockSize.y);

    fused_preprocess_gray_stereo_kernel<<<gridSize, blockSize, 0, stream>>>(
        img_left.ptr<uint8_t>(),
        static_cast<int>(img_left.step),
        img_left.channels(),
        img_right.ptr<uint8_t>(),
        static_cast<int>(img_right.step),
        img_right.channels(),
        img_left.rows, img_left.cols,
        output, target_height, target_width,
        pad_x, pad_y,
        new_w, new_h,
        fused_inv_scale(img_left.cols, new_w),
        fused_inv_scale(img_left.rows, new_h));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchFusedPreprocess_Gray_Stereo: %s\n", cudaGetErrorString(err));
    }
}

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
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((new_w + blockSize.x - 1) / blockSize.x,
                  (new_h + blockSize.y - 1) / blockSize.y);

    fused_preprocess_rgb_stereo_kernel<<<gridSize, blockSize, 0, stream>>>(
        img_left.ptr<uint8_t>(),
        static_cast<int>(img_left.step),
        img_left.channels(),
        img_right.ptr<uint8_t>(),
        static_cast<int>(img_right.step),
        img_right.channels(),
        img_left.rows, img_left.cols,
        output, target_height, target_width,
        pad_x, pad_y,
        new_w, new_h,
        fused_inv_scale(img_left.cols, new_w),
        fused_inv_scale(img_left.rows, new_h));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchFusedPreprocess_RGB_Stereo: %s\n", cudaGetErrorString(err));
    }
}

__global__ void normalize_keypoints_to_model_space_kernel(
    const float *kps_px,
    float *kps_model,
    int N,
    float fwd_scale,
    float x_offset,
    float y_offset,
    int model_w,
    int model_h)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const float px_x = kps_px[idx * 2 + 0];
    const float px_y = kps_px[idx * 2 + 1];

    const float model_x = px_x * fwd_scale + x_offset;
    const float model_y = px_y * fwd_scale + y_offset;

    kps_model[idx * 2 + 0] = (model_x / static_cast<float>(model_w)) * 2.0f - 1.0f;
    kps_model[idx * 2 + 1] = (model_y / static_cast<float>(model_h)) * 2.0f - 1.0f;
}

void launchNormalizeKeypointsToModelSpace(
    const float *d_kps_px,
    float *d_kps_model,
    int N,
    float fwd_scale,
    float x_offset,
    float y_offset,
    int model_w,
    int model_h,
    cudaStream_t stream)
{
    if (N <= 0 || d_kps_px == nullptr || d_kps_model == nullptr)
        return;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    normalize_keypoints_to_model_space_kernel<<<blocks, threads, 0, stream>>>(
        d_kps_px, d_kps_model, N, fwd_scale, x_offset, y_offset, model_w, model_h);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchNormalizeKeypointsToModelSpace: %s\n", cudaGetErrorString(err));
    }
}

__global__ void normalize_keypoints_lightglue_kernel(
    const float *kps_px,
    float *kps_norm,
    int n_use,
    int max_kp,
    int kpt_dim,
    float cx,
    float cy,
    float inv_scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_kp)
        return;

    if (idx < n_use)
    {
        kps_norm[idx * kpt_dim + 0] = (kps_px[idx * kpt_dim + 0] - cx) * inv_scale;
        kps_norm[idx * kpt_dim + 1] = (kps_px[idx * kpt_dim + 1] - cy) * inv_scale;
        if (kpt_dim >= 4)
        {
            kps_norm[idx * kpt_dim + 2] = kps_px[idx * kpt_dim + 2];
            kps_norm[idx * kpt_dim + 3] = kps_px[idx * kpt_dim + 3];
        }
    }
    else
    {
        for (int j = 0; j < kpt_dim; ++j)
            kps_norm[idx * kpt_dim + j] = 0.0f;
    }
}

void launchNormalizeKeypointsLightGlue(
    const float *d_kps_px,
    float *d_kps_norm,
    int n_use,
    int max_kp,
    int kpt_dim,
    float cx,
    float cy,
    float inv_scale,
    cudaStream_t stream)
{
    if (max_kp <= 0 || d_kps_norm == nullptr || kpt_dim <= 0)
        return;
    if (n_use < 0)
        n_use = 0;
    if (n_use > max_kp)
        n_use = max_kp;
    const int threads = 256;
    const int blocks = (max_kp + threads - 1) / threads;
    normalize_keypoints_lightglue_kernel<<<blocks, threads, 0, stream>>>(
        d_kps_px, d_kps_norm, n_use, max_kp, kpt_dim, cx, cy, inv_scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchNormalizeKeypointsLightGlue: %s\n", cudaGetErrorString(err));
    }
}

__global__ void gather_kpts_desc_kernel(
    const float *kps_src,
    const float *desc_src,
    const int *kept_indices,
    int n_kept,
    int D,
    int kpt_stride,
    float *kps_dst,
    float *desc_dst)
{
    const int row = blockIdx.x;
    if (row >= n_kept)
        return;

    const int src_row = kept_indices[row];

    // Each thread in the block copies a chunk of the descriptor row.
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const float *src_d = desc_src + static_cast<size_t>(src_row) * D;
    float *dst_d = desc_dst + static_cast<size_t>(row) * D;
    for (int j = tid; j < D; j += stride)
        dst_d[j] = src_d[j];

    for (int j = tid; j < kpt_stride; j += stride)
        kps_dst[row * kpt_stride + j] = kps_src[src_row * kpt_stride + j];
}

void launchGatherKeypointsAndDescriptors(
    const float *d_kps_src,
    const float *d_desc_src,
    const int *d_kept_indices,
    int n_kept,
    int D,
    int kpt_stride,
    float *d_kps_dst,
    float *d_desc_dst,
    cudaStream_t stream)
{
    if (n_kept <= 0 || D <= 0 || kpt_stride <= 0 ||
        d_kps_src == nullptr || d_desc_src == nullptr ||
        d_kept_indices == nullptr || d_kps_dst == nullptr || d_desc_dst == nullptr)
        return;

    // One block per kept row, threads copy the descriptor across the D axis.
    const int threads = (D < 128) ? D : 128;
    gather_kpts_desc_kernel<<<n_kept, threads, 0, stream>>>(
        d_kps_src, d_desc_src, d_kept_indices, n_kept, D, kpt_stride, d_kps_dst, d_desc_dst);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchGatherKeypointsAndDescriptors: %s\n", cudaGetErrorString(err));
    }
}

// ALIKED dense-pipeline DKD postproc
// ----------------------------------------------------------------------------
// Replicates the in-engine DKD (soft_detect.py:DKD._detect_keypoints_topk_batched)
// outside of TensorRT, so the dense extractor engine can be a fully fused
// backbone+score_head with better performance at B=2.
//
// Three stages, all on `stream`:
//   1) aliked_nms_score_kernel: 5x5 strict-max NMS, border zeroing.
//      Output: nms_scores (B, H*W) FP32 — keys for top-K.
//   2) cub::DeviceSegmentedRadixSort sorts (score, flat_index) per batch,
//      descending. Top max_K become candidates.
//   3) aliked_subpixel_kernel: per (b, k) 5x5 soft-argmax (T=0.1) on FP32 score
//      patch, bilinear-sample refined score, emit
//        kpts_n (normalised [-1,1]), kpts_px (pixel coords post-letterbox),
//        scores, num_kpts.

namespace
{

    constexpr int ALIKED_PATCH = 5; // kernel_size = 2 * radius + 1
    constexpr int ALIKED_PATCH_SQ = 25;
    constexpr int ALIKED_RADIUS = 2;

    __device__ __forceinline__ uint64_t aliked_pack_key(float score, int flat)
    {
        const uint32_t s_bits = __float_as_uint(score);
        const uint32_t f_bits = ~static_cast<uint32_t>(flat);
        return (static_cast<uint64_t>(s_bits) << 32) | static_cast<uint64_t>(f_bits);
    }

    __device__ __forceinline__ float aliked_unpack_score(uint64_t key)
    {
        return __uint_as_float(static_cast<uint32_t>(key >> 32));
    }

    __device__ __forceinline__ int aliked_unpack_flat(uint64_t key)
    {
        return static_cast<int>(~static_cast<uint32_t>(key));
    }

    // Pass 1: NMS + COMPACT into a per-batch candidate buffer.
    __global__ void aliked_nms_compact_kernel(
        const float *__restrict__ score_map, // (B, 1, H, W)
        int B, int H, int W,
        int radius, float scores_th,
        int max_cand,                     // per-batch capacity
        int *__restrict__ counters,       // (B,) atomic claim
        uint64_t *__restrict__ cand_keys) // (B, max_cand) — packed (score, ~flat)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int b = blockIdx.z;
        if (x >= W || y >= H || b >= B)
            return;

        // Border strip
        if (x < radius || x >= W - radius || y < radius || y >= H - radius)
            return;

        const int flat = y * W + x;
        const float *sm = score_map + (size_t)b * H * W;
        const float s = sm[flat];
        if (s <= scores_th)
            return;

        // 5x5 strict-max check. Tie-break: lower flat index wins (deterministic
        // matching of cuDNN max_pool2d's "first equal value" behaviour).
        bool is_max = true;
#pragma unroll
        for (int dy = -ALIKED_RADIUS; dy <= ALIKED_RADIUS; ++dy)
        {
#pragma unroll
            for (int dx = -ALIKED_RADIUS; dx <= ALIKED_RADIUS; ++dx)
            {
                if (dx == 0 && dy == 0)
                    continue;
                const int nflat = (y + dy) * W + (x + dx);
                const float v = sm[nflat];
                if (v > s)
                {
                    is_max = false;
                }
                else if (v == s && nflat < flat)
                {
                    is_max = false;
                }
            }
        }
        if (!is_max)
            return;

        const int slot = atomicAdd(&counters[b], 1);
        if (slot < max_cand)
        {
            cand_keys[(size_t)b * max_cand + slot] = aliked_pack_key(s, flat);
        }
    }

    // Pass 3: per (b, k) 5x5 soft-argmax + bilinear score sample + letterbox inverse.
    __global__ void aliked_subpixel_kernel(
        const float *__restrict__ score_map,      // (B, 1, H, W)
        const uint64_t *__restrict__ sorted_keys, // (B, max_cand) — packed (score, ~flat); top entries first
        int B, int H, int W, int K, int max_cand,
        float temperature,
        float scale, float x_off, float y_off,
        float *__restrict__ kpts_n,  // (B, K, 2) — normalised [-1, 1]
        float *__restrict__ kpts_px, // (B, K, 2) — pixel coords (letterbox-inverted)
        float *__restrict__ scores,  // (B, K)
        int *__restrict__ num_kpts)  // (B,)
    {
        const int b = blockIdx.y;
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= B || k >= K)
            return;

        const uint64_t key = sorted_keys[(size_t)b * max_cand + k];
        const float kpt_score_key = aliked_unpack_score(key);

        // Pad zero-keypoints when fewer than K survived NMS.
        if (kpt_score_key <= 0.f)
        {
            kpts_n[((size_t)b * K + k) * 2 + 0] = 0.f;
            kpts_n[((size_t)b * K + k) * 2 + 1] = 0.f;
            kpts_px[((size_t)b * K + k) * 2 + 0] = 0.f;
            kpts_px[((size_t)b * K + k) * 2 + 1] = 0.f;
            scores[(size_t)b * K + k] = 0.f;
            if (k == K - 1)
            {
                // Find num_kpts: count of non-zero scores in sorted_keys prefix.
                int count = 0;
                for (int i = 0; i < K; ++i)
                {
                    if (aliked_unpack_score(sorted_keys[(size_t)b * max_cand + i]) > 0.f)
                        ++count;
                    else
                        break;
                }
                num_kpts[b] = count;
            }
            return;
        }

        if (k == K - 1)
        {
            num_kpts[b] = K;
        }

        const int flat = aliked_unpack_flat(key);
        const int y = flat / W;
        const int x = flat - y * W;

        const float *sm = score_map + (size_t)b * H * W;

        // Load 5x5 patch around (y, x). Border-safe (NMS already excluded < radius
        // strip but defensively clamp anyway).
        float patch[ALIKED_PATCH_SQ];
        float max_v = -1e30f;
#pragma unroll
        for (int dy = -ALIKED_RADIUS; dy <= ALIKED_RADIUS; ++dy)
        {
#pragma unroll
            for (int dx = -ALIKED_RADIUS; dx <= ALIKED_RADIUS; ++dx)
            {
                const int yy = y + dy;
                const int xx = x + dx;
                float v = 0.f;
                if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                {
                    v = sm[yy * W + xx];
                }
                patch[(dy + ALIKED_RADIUS) * ALIKED_PATCH + (dx + ALIKED_RADIUS)] = v;
                if (v > max_v)
                    max_v = v;
            }
        }

        // Soft-argmax: w[k] = exp((s[k] - max) / T); residual = sum(w * (dx, dy)) / sum(w).
        const float inv_T = 1.f / temperature;
        float sum_w = 0.f, res_x = 0.f, res_y = 0.f;
#pragma unroll
        for (int slot = 0; slot < ALIKED_PATCH_SQ; ++slot)
        {
            const float w = expf((patch[slot] - max_v) * inv_T);
            sum_w += w;
            const int dy = slot / ALIKED_PATCH - ALIKED_RADIUS;
            const int dx = slot - (slot / ALIKED_PATCH) * ALIKED_PATCH - ALIKED_RADIUS;
            res_x += w * (float)dx;
            res_y += w * (float)dy;
        }
        const float inv_sum = 1.f / sum_w;
        res_x *= inv_sum;
        res_y *= inv_sum;

        const float refined_x = (float)x + res_x;
        const float refined_y = (float)y + res_y;

        // Bilinear-sample score_map at refined position (matches DKD's grid_sample
        // align_corners=True semantics).
        const int x0 = max(0, min(W - 1, (int)floorf(refined_x)));
        const int x1 = max(0, min(W - 1, x0 + 1));
        const int y0 = max(0, min(H - 1, (int)floorf(refined_y)));
        const int y1 = max(0, min(H - 1, y0 + 1));
        const float fx = refined_x - (float)x0;
        const float fy = refined_y - (float)y0;
        const float v00 = sm[y0 * W + x0];
        const float v01 = sm[y0 * W + x1];
        const float v10 = sm[y1 * W + x0];
        const float v11 = sm[y1 * W + x1];
        const float v = (1.f - fy) * ((1.f - fx) * v00 + fx * v01) + fy * ((1.f - fx) * v10 + fx * v11);

        scores[(size_t)b * K + k] = v;

        // Normalised coords for dhead engine: 2 * pos / (size - 1) - 1
        kpts_n[((size_t)b * K + k) * 2 + 0] = refined_x / (float)(W - 1) * 2.f - 1.f;
        kpts_n[((size_t)b * K + k) * 2 + 1] = refined_y / (float)(H - 1) * 2.f - 1.f;

        // Pixel coords in original image (apply letterbox inverse).
        kpts_px[((size_t)b * K + k) * 2 + 0] = (refined_x - x_off) * scale;
        kpts_px[((size_t)b * K + k) * 2 + 1] = (refined_y - y_off) * scale;
    }

    // Per-batch candidate capacity. Theoretical max under 5x5 strict NMS is HW/25;
    // HW/8 gives a generous safety margin without ballooning sort cost.
    __host__ __device__ inline int aliked_dkd_max_cand(int H, int W)
    {
        return (H * W + 7) / 8;
    }

    // Workspace layout (all device pointers, contiguous after alignment):
    //   counters       : int     [B]                  atomic claim
    //   cand_keys_in   : uint64  [B * max_cand]       packed (score, ~flat); zero-init
    //   cand_keys_out  : uint64  [B * max_cand]       cub sort destination
    //   segment_offsets: int     [B + 1]              [0, max_cand, 2*max_cand, ...]
    //   cub temp       : variable
    struct AlikedDkdWorkspace
    {
        int *counters;
        uint64_t *cand_keys_in;
        uint64_t *cand_keys_out;
        int *segment_offsets;
        void *cub_temp;
        size_t cub_temp_bytes;
        int max_cand;
    };

    inline size_t roundUp(size_t x, size_t a) { return (x + a - 1) / a * a; }

    struct AlikedDkdLayout
    {
        size_t off_counters;
        size_t off_keys_in, off_keys_out, off_seg, off_cub;
        size_t size_cub;
        size_t total;
        int max_cand;
    };

    static AlikedDkdLayout aliked_dkd_layout(int B, int H, int W)
    {
        AlikedDkdLayout L{};
        const int max_cand = aliked_dkd_max_cand(H, W);
        L.max_cand = max_cand;
        const size_t key_bytes = sizeof(uint64_t) * B * max_cand;
        const size_t seg_bytes = sizeof(int) * (B + 1);
        const size_t counter_bytes = sizeof(int) * B;

        // Probe cub workspace size.
        // Cast to int* so cub instantiates the int*-pointer path
        // (no kernel is launched while cub_temp is nullptr).
        size_t cub_temp = 0;
        cub::DeviceSegmentedRadixSort::SortKeysDescending(
            nullptr, cub_temp,
            static_cast<const uint64_t *>(nullptr), static_cast<uint64_t *>(nullptr),
            static_cast<int>(B * max_cand), B,
            static_cast<const int *>(nullptr), static_cast<const int *>(nullptr));
        L.size_cub = cub_temp;

        constexpr size_t A = 256;
        L.off_counters = 0;
        L.off_keys_in = roundUp(L.off_counters + counter_bytes, A);
        L.off_keys_out = roundUp(L.off_keys_in + key_bytes, A);
        L.off_seg = roundUp(L.off_keys_out + key_bytes, A);
        L.off_cub = roundUp(L.off_seg + seg_bytes, A);
        L.total = roundUp(L.off_cub + L.size_cub, A);
        return L;
    }

    __global__ void fill_segment_offsets_kernel(int *segs, int B, int max_cand)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i <= B)
            segs[i] = i * max_cand;
    }

} // anonymous namespace

size_t alikedDkdPostprocWorkspaceBytes(int B, int H, int W, int /*max_K*/)
{
    return aliked_dkd_layout(B, H, W).total;
}

void launchAlikedDkdPostproc(
    const float *d_score_map,
    int B, int H, int W,
    int max_K,
    int nms_radius,
    float scores_th,
    float subpixel_temperature,
    float scale,
    float x_offset,
    float y_offset,
    float *d_kpts_px,
    float *d_kpts_n,
    float *d_scores,
    int *d_num_kpts,
    void *d_workspace,
    size_t workspace_bytes,
    cudaStream_t stream)
{
    if (B <= 0 || H <= 0 || W <= 0 || max_K <= 0 || d_score_map == nullptr ||
        d_workspace == nullptr)
    {
        return;
    }
    const AlikedDkdLayout L = aliked_dkd_layout(B, H, W);
    if (workspace_bytes < L.total)
    {
        printf("launchAlikedDkdPostproc: workspace too small (%zu < %zu)\n",
               workspace_bytes, L.total);
        return;
    }

    (void)nms_radius;

    char *base = static_cast<char *>(d_workspace);
    AlikedDkdWorkspace ws{};
    ws.counters = reinterpret_cast<int *>(base + L.off_counters);
    ws.cand_keys_in = reinterpret_cast<uint64_t *>(base + L.off_keys_in);
    ws.cand_keys_out = reinterpret_cast<uint64_t *>(base + L.off_keys_out);
    ws.segment_offsets = reinterpret_cast<int *>(base + L.off_seg);
    ws.cub_temp = reinterpret_cast<void *>(base + L.off_cub);
    ws.cub_temp_bytes = L.size_cub;
    ws.max_cand = L.max_cand;

    // Pre-zero the candidate slabs and counters so unused slots have key 0
    // (sort to the bottom) and counters start fresh.
    cudaMemsetAsync(ws.counters, 0,
                    sizeof(int) * B, stream);
    cudaMemsetAsync(ws.cand_keys_in, 0,
                    sizeof(uint64_t) * B * ws.max_cand, stream);

    // Initialise segment offsets [0, max_cand, 2*max_cand, ..., B*max_cand].
    {
        const int threads = 32;
        const int blocks = (B + 1 + threads - 1) / threads;
        fill_segment_offsets_kernel<<<blocks, threads, 0, stream>>>(
            ws.segment_offsets, B, ws.max_cand);
    }

    // Pass 1: NMS + atomic compact into per-batch candidate slab.
    {
        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x,
                  (H + block.y - 1) / block.y,
                  B);
        aliked_nms_compact_kernel<<<grid, block, 0, stream>>>(
            d_score_map, B, H, W, ALIKED_RADIUS, scores_th,
            ws.max_cand, ws.counters, ws.cand_keys_in);
    }

    // Pass 2: cub segmented radix sort over the SMALL candidate buffer
    // (~HW/8 per batch instead of HW). Top max_K entries land at the front.
    // The 64-bit composite key sorts (score desc, flat asc) as a total order,
    // so equal-score keypoints come out in a deterministic position.
    {
        size_t temp_bytes = ws.cub_temp_bytes;
        cub::DeviceSegmentedRadixSort::SortKeysDescending(
            ws.cub_temp, temp_bytes,
            ws.cand_keys_in, ws.cand_keys_out,
            B * ws.max_cand, B,
            ws.segment_offsets, ws.segment_offsets + 1,
            0, sizeof(uint64_t) * 8,
            stream);
    }

    // Pass 3: subpixel + bilinear score + letterbox inverse + zero-pad.
    {
        const int threads = 64;
        dim3 block(threads);
        dim3 grid((max_K + threads - 1) / threads, B);
        aliked_subpixel_kernel<<<grid, block, 0, stream>>>(
            d_score_map,
            ws.cand_keys_out,
            B, H, W, max_K, ws.max_cand,
            subpixel_temperature,
            scale, x_offset, y_offset,
            d_kpts_n, d_kpts_px, d_scores, d_num_kpts);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in launchAlikedDkdPostproc: %s\n", cudaGetErrorString(err));
    }
}