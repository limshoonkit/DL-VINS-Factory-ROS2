#ifndef LOOP_FUSION_DINO_VPR_HPP
#define LOOP_FUSION_DINO_VPR_HPP

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "loop_fusion/vlad.hpp"

namespace uosm::loop_fusion
{
    // AnyLoc universal loop-closure head: DINOv2 ViT-S/14 (frozen, TensorRT) +
    // VLAD over the dense patch tokens against a fixed vocabulary. Produces a
    // single L2-normalised global descriptor per keyframe image.
    //
    // Artifacts are exported offline by AnyLoc/vpr/dino_vpr_export.py:
    //   engine_path : strict fixed-resolution .engine (input 1x3xHxW, output
    //                 tokens 1xTx384 with T = (H/14)*(W/14))
    //   vocab_path  : VladCodebook .bin (int32 k | int32 384 | k*384 float32)
    class DinoVpr
    {
    public:
        struct Params
        {
            std::string engine_path;
            std::string vocab_path;
            int input_h = 280; // multiple of 14
            int input_w = 448; // multiple of 14
            int embed_dim = 384;
            int patch = 14;
        };

        explicit DinoVpr(const Params &p);
        ~DinoVpr();

        DinoVpr(const DinoVpr &) = delete;
        DinoVpr &operator=(const DinoVpr &) = delete;

        bool ok() const
        {
            return engine_ && context_ && d_input_ && d_tokens_ && vlad_.ready();
        }
        int vladDim() const { return vlad_.vladDim(); }

        // image: CV_8UC1 (mono) or CV_8UC3 (BGR). Returns the L2-normalised
        // VLAD global descriptor (vladDim()), or an empty vector on failure.
        std::vector<float> compute(const cv::Mat &image);

    private:
        bool initEngine();
        bool discoverNames();
        bool allocate();

        Params params_;
        int num_tokens_ = 0;

        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<nvinfer1::IExecutionContext> context_;
        cudaStream_t stream_ = nullptr;

        std::string input_name_;
        std::string output_name_;

        float *d_input_ = nullptr;  // 1 x 3 x H x W
        float *d_tokens_ = nullptr; // 1 x T x embed_dim
        std::vector<float> h_tokens_;

        VladCodebook vlad_;
    };
} // namespace uosm::loop_fusion

#endif // LOOP_FUSION_DINO_VPR_HPP
