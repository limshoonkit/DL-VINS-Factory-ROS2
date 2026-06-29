#include "loop_fusion/dino_vpr_preprocess.cuh"

#include <cstdint>
#include <cstdio>

namespace uosm::loop_fusion
{
    namespace
    {
        // src->dst inverse scale: a destination pixel of size 1 maps to
        // (src / dst) source pixels.
        __host__ __device__ __forceinline__ float inv_scale(int src, int dst)
        {
            return static_cast<float>(src) / static_cast<float>(dst);
        }

        __device__ __forceinline__ void bilinear_sample(
            const uint8_t *src, int src_step, int src_channels,
            int src_h, int src_w, float sx, float sy,
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

        __global__ void dino_preprocess_kernel(
            const uint8_t *src, int src_step, int src_channels,
            int src_h, int src_w,
            float *dst, int dst_h, int dst_w,
            int pad_x, int pad_y, int new_w, int new_h,
            float inv_scale_x, float inv_scale_y)
        {
            const int tx = blockIdx.x * blockDim.x + threadIdx.x;
            const int ty = blockIdx.y * blockDim.y + threadIdx.y;
            if (tx >= dst_w || ty >= dst_h)
                return;

            // ImageNet normalization constants — MUST match
            // AnyLoc/vpr/dino_vpr_export.py IMAGENET_MEAN / IMAGENET_STD.
            const float mean[3] = {0.485f, 0.456f, 0.406f};
            const float inv_std[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

            float r = 0.0f, g = 0.0f, b = 0.0f; // pre-/255 values (0 in pad region)
            const bool inside = (tx >= pad_x && tx < pad_x + new_w &&
                                 ty >= pad_y && ty < pad_y + new_h);
            if (inside)
            {
                const float sx = (tx - pad_x + 0.5f) * inv_scale_x - 0.5f;
                const float sy = (ty - pad_y + 0.5f) * inv_scale_y - 0.5f;
                float c0, c1, c2;
                bilinear_sample(src, src_step, src_channels, src_h, src_w,
                                sx, sy, c0, c1, c2);
                // Source is BGR (3ch) or gray-broadcast (1ch); output is RGB order.
                r = (src_channels == 1) ? c0 : c2;
                g = (src_channels == 1) ? c0 : c1;
                b = c0;
            }

            const int chan = dst_h * dst_w;
            const int idx = ty * dst_w + tx;
            const float inv_255 = 1.0f / 255.0f;
            dst[idx] = (r * inv_255 - mean[0]) * inv_std[0];
            dst[chan + idx] = (g * inv_255 - mean[1]) * inv_std[1];
            dst[2 * chan + idx] = (b * inv_255 - mean[2]) * inv_std[2];
        }
    } // namespace

    void launchDinoPreprocess(
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
        const dim3 block(16, 16);
        const dim3 grid((target_width + block.x - 1) / block.x,
                        (target_height + block.y - 1) / block.y);

        dino_preprocess_kernel<<<grid, block, 0, stream>>>(
            img.ptr<uint8_t>(),
            static_cast<int>(img.step),
            img.channels(),
            img.rows, img.cols,
            output, target_height, target_width,
            pad_x, pad_y, new_w, new_h,
            inv_scale(img.cols, new_w),
            inv_scale(img.rows, new_h));

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA launch error in launchDinoPreprocess: %s\n",
                   cudaGetErrorString(err));
    }
} // namespace uosm::loop_fusion
