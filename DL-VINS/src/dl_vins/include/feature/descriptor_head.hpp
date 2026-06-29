#ifndef DESCRIPTOR_HEAD_HPP_
#define DESCRIPTOR_HEAD_HPP_

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "utility/trt_common.hpp"

#include <memory>
#include <string>

namespace uosm
{
    namespace perception
    {
        // Runs the ALIKED desc-head-only TensorRT engine that takes a pre-computed
        // dense feature_map (from the DenseExtractor) and external keypoints, and
        // emits SDDH descriptors. Inputs the dense feature_map directly (no
        // internal backbone re-run) and supports external CUDA streams for
        // GPU-resident chaining with the dense extractor + DKD postproc.
        //
        //   inputs:
        //     feature_map : (B, dim, H, W) float32  (dim = 128 for aliked-n16)
        //     kpts_n      : (B, K, 2)      float32  normalised to [-1, 1]
        //   outputs:
        //     descriptors : (B, K, 128)    float32  L2-normalised
        class DescriptorHead
        {
        public:
            struct Params
            {
                std::string engine_path;
                int max_kp = 256;
                int model_h = 480;
                int model_w = 768;
                int feature_dim = 128;
                int batch = 2;
            };

            explicit DescriptorHead(Params p);
            ~DescriptorHead();

            bool is_initialized() const
            {
                return (engine_ != nullptr) && (context_ != nullptr) && (stream_ != nullptr);
            }
            int getDescriptorDim() const { return desc_dim_; }
            int getMaxKp() const { return params_.max_kp; }
            int getBatch() const { return params_.batch; }
            int getModelH() const { return params_.model_h; }
            int getModelW() const { return params_.model_w; }
            int getFeatureDim() const { return params_.feature_dim; }

            // Chain the engine into a caller-owned CUDA stream so dense -> CUDA postproc -> dhead
            // stays GPU-resident without host round-trips. The caller is responsible for any
            // downstream synchronisation.
            bool runInference(const float *d_feature_map,
                              const float *d_kpts_n,
                              cudaStream_t external_stream);

            bool runInference(const float *d_feature_map, const float *d_kpts_n);

            void *getDescriptorsGpu() const { return d_descriptors_; }

        private:
            bool initTensorRT();
            void discoverTensorNames();
            void allocateBuffers();
            void setupBindings();
            bool warmup(int n_iters = 3);

            Params params_;

            std::unique_ptr<nvinfer1::IRuntime> runtime_;
            std::unique_ptr<nvinfer1::ICudaEngine> engine_;
            std::unique_ptr<nvinfer1::IExecutionContext> context_;
            cudaStream_t stream_ = nullptr;

            std::string feature_map_name_;
            std::string kpts_name_;
            std::string desc_name_;
            int desc_dim_ = 128;
            float *d_descriptors_ = nullptr;
            size_t descriptors_bytes_ = 0;
        };
    } // namespace perception
} // namespace uosm

#endif // DESCRIPTOR_HEAD_HPP_
