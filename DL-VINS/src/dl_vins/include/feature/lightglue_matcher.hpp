#ifndef LIGHTGLUE_MATCHER_HPP_
#define LIGHTGLUE_MATCHER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "descriptor_matcher.hpp"
#include "utility/trt_common.hpp" // shared DynamicOutputAllocator + TrtLogger

namespace uosm
{
    namespace perception
    {

        // Minimal TensorRT wrapper around a LightGlue engine.
        //
        // Expected engine I/O (typical LightGlue-ONNX-Jetson export):
        //   Inputs:
        //     - "kpts0" : (1, N0, 2) float  -- pixel coords (will be normalized to [-1, 1])
        //     - "kpts1" : (1, N1, 2) float
        //     - "desc0" : (1, N0, D) float  -- descriptor per keypoint
        //     - "desc1" : (1, N1, D) float
        //   Outputs (names auto-discovered):
        //     - "matches" : (M, 2) int64  -- rows of [idx0, idx1]
        //     - "mscores" : (M,)    float -- match scores
        class LightGlueMatcher : public DescriptorMatcher
        {
        public:
            using Params = DescriptorMatcherParams;

            explicit LightGlueMatcher(Params params);
            ~LightGlueMatcher() override;

            bool is_initialized() const override
            {
                return (engine_ != nullptr) && (slot_a_.context != nullptr) && (slot_a_.stream != nullptr);
            }

            int getDescriptorDim() const override { return params_.descriptor_dim; }
            int getMaxKeypoints() const override { return params_.max_keypoints; }
            const std::string &name() const override { return kName; }

            // Returns the list of matches whose softmax score >= params_.score_threshold.
            std::vector<DescriptorMatch> match(const float *d_kpts0_px, int n0,
                                                  const void *d_desc0,
                                                  const float *d_kpts1_px, int n1,
                                                  const void *d_desc1,
                                                  int image_h, int image_w) override;

            // Batched matching. Requires a batched ("_b{N}") engine: all pairs run in
            // one TRT inference using slot_a_'s context. Non-batched engines fall back
            // to a per-pair match() loop on slot_a_.
            std::vector<std::vector<DescriptorMatch>>
            matchBatched(const std::vector<MatchInputs> &pairs) override;

        private:
            // One TRT execution slot: context + stream + scratch kpts buffers + output
            // allocators. match() and matchBatched() share a single slot
            // (slot_a_) — concurrent two-pair overlap across slots was removed once
            // all production engines were known to be batched.
            struct MatcherSlot
            {
                std::unique_ptr<nvinfer1::IExecutionContext> context;
                std::unordered_map<std::string,
                                   std::unique_ptr<uosm::utility::DynamicOutputAllocator>>
                    allocators;
                cudaStream_t stream = nullptr;
                float *d_kpts0 = nullptr;
                float *d_kpts1 = nullptr;
                size_t d_kpts0_capacity_bytes = 0;
                size_t d_kpts1_capacity_bytes = 0;
            };

            static const std::string kName;
            Params params_;

            std::unique_ptr<nvinfer1::IRuntime> runtime_;
            std::unique_ptr<nvinfer1::ICudaEngine> engine_;

            // Single execution slot. Used by both match() (single pair) and the
            // batched-engine path of matchBatched().
            MatcherSlot slot_a_;

            // I/O tensor names discovered at init.
            std::string kpts0_name_, kpts1_name_, desc0_name_, desc1_name_;
            std::string matches_name_, mscores_name_;

            // Batched-engine state. batched_ is true when the engine's matches output is
            // (M,3).
            bool batched_ = false;
            int batch_capacity_ = 1;
            int kpt_dim_ = 2;
            float *d_batch_kpts0_ = nullptr;
            float *d_batch_kpts1_ = nullptr;
            void *d_batch_desc0_ = nullptr;
            void *d_batch_desc1_ = nullptr;

            bool init();
            void shutdown();
            bool discoverTensorNames();
            void detectBatched();
            bool initSlot(MatcherSlot &slot);
            void shutdownSlot(MatcherSlot &slot);
            bool ensureSlotCapacity(MatcherSlot &slot);
            bool ensureBatchBuffers();
            std::vector<DescriptorMatch> runPairOnSlot(MatcherSlot &slot, const MatchInputs &in);
            bool enqueuePairAsync(MatcherSlot &slot, const MatchInputs &in,
                             int &n0_use_out, int &n1_use_out);
            std::vector<DescriptorMatch> collectPairAfterSync(MatcherSlot &slot, int n0_use, int n1_use);
            bool warmup(int n_iters = 3);
        };

    } // namespace perception
} // namespace uosm

#endif // LIGHTGLUE_MATCHER_HPP_
