#ifndef DENSE_EXTRACTOR_HPP_
#define DENSE_EXTRACTOR_HPP_

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <opencv2/core/cuda.hpp>

#include "utility/trt_common.hpp"

#include <memory>
#include <string>

namespace uosm
{
    namespace perception
    {
        // Runs the ALIKED dense-headless extractor TensorRT engine: backbone +
        // score_head only, no DKD. Outputs the dense `score_map` and
        // L2-normalised `feature_map` tensors that feed the custom CUDA DKD
        // postproc and the desc-head-only engine.
        //
        //   inputs:
        //     image       : (B, 3, H, W) float32, RGB in [0, 1]
        //   outputs:
        //     score_map   : (B, 1, H, W)
        //     feature_map : (B, 128, H, W)
        //
        class DenseExtractor
        {
        public:
            struct Params
            {
                std::string engine_path;
                int batch_size = 2;
                int input_h = 480;
                int input_w = 768;
                int input_channels = 3;
                int feature_dim = 128;
                bool profile_inference = false;
            };

            explicit DenseExtractor(Params p);
            ~DenseExtractor();

            bool is_initialized() const
            {
                return (engine_ != nullptr) && (context_ != nullptr);
            }
            int getBatch() const { return params_.batch_size; }
            int getInputH() const { return params_.input_h; }
            int getInputW() const { return params_.input_w; }
            int getFeatureDim() const { return params_.feature_dim; }

            // Letterbox parameters from the most recent run (for downstream
            // letterbox-inverse in the CUDA DKD postproc kernel).
            float getScale() const { return scale_; }
            float getXOffset() const { return x_offset_; }
            float getYOffset() const { return y_offset_; }

            // Preprocess (BGR uint8) -> NCHW (RGB float32) and run the dense
            // engine on `stream`. 
            bool runInferenceStereoFromGPU(const cv::cuda::GpuMat &gpu_img_left,
                                           const cv::cuda::GpuMat &gpu_img_right,
                                           cudaStream_t stream);
            bool runInferenceFromGPU(const cv::cuda::GpuMat &gpu_img,
                                     cudaStream_t stream);

            const float *getScoreMapGpu() const { return d_score_map_; }
            const float *getFeatureMapGpu() const { return d_feature_map_; }

        private:
            bool initTensorRT();
            void discoverTensorNames();
            bool allocateBuffers();
            void setupBindings();
            bool warmup(int n_iters = 3);

            Params params_;

            std::unique_ptr<nvinfer1::IRuntime> runtime_;
            std::unique_ptr<nvinfer1::ICudaEngine> engine_;
            std::unique_ptr<nvinfer1::IExecutionContext> context_;
            cudaStream_t internal_stream_ = nullptr; // used only for warmup

            std::string image_name_;
            std::string score_map_name_;
            std::string feature_map_name_;

            float *d_input_ = nullptr;        // (B, 3, H, W)
            float *d_score_map_ = nullptr;    // (B, 1, H, W)
            float *d_feature_map_ = nullptr;  // (B, 128, H, W)

            float scale_ = 1.0f;
            float x_offset_ = 0.0f;
            float y_offset_ = 0.0f;
        };
    } // namespace perception
} // namespace uosm

#endif // DENSE_EXTRACTOR_HPP_
