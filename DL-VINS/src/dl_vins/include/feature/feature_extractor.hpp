#ifndef FEATURE_EXTRACTOR_HPP_
#define FEATURE_EXTRACTOR_HPP_

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <array>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <cstdint>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <nvtx3/nvToolsExt.h> // For profiling with NVTX
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include "type_conv_helper.cuh"
#include "utility/trt_common.hpp"

namespace uosm
{
    namespace perception
    {
        class FeatureExtractor
        {
        public:
            struct FeatureExtractorParams
            {
                std::string engine_path;
                int input_width;
                int input_height;
                int max_keypoints;
                bool profile_inference;
                bool use_unified_memory;
                std::string model_type;
                int input_channels;
                int batch_size = 1U;
            };

            // Stereo output structure (batch=2 results)
            struct StereoFeatureData
            {
                std::vector<cv::Point2f> keypoints_left;
                std::vector<cv::Point2f> keypoints_right;
                std::vector<float> scores_left;
                std::vector<float> scores_right;
                std::vector<std::array<float, 4>> cov_left;
                std::vector<std::array<float, 4>> cov_right;
                int num_keypoints_left = 0U;
                int num_keypoints_right = 0U;
            };

            struct FeatureExtractionData
            {
                void *h_pinned_buffer_; // Single large pinned buffer
                size_t total_output_size_ = 0U;
                // Offsets within the pinned buffer
                size_t num_kpts_offset_ = 0U;
                size_t keypoints_offset_ = 0U;
                size_t scores_offset_ = 0U;
                size_t descriptors_offset_ = 0U;
                size_t covariances_offset_ = 0U;
                int descriptor_dim_ = 0U;
                // Letterbox transform params (for any aspect ratio)
                float scale_ = 1.0f;    // Uniform scale factor (same for x and y)
                float x_offset_ = 0.0f; // Padding offset in x (pixels in model input space)
                float y_offset_ = 0.0f; // Padding offset in y (pixels in model input space)
            };

            FeatureExtractor(FeatureExtractorParams params);
            ~FeatureExtractor();

            // Shared TRT logger
            using TrtLogger = uosm::utility::TrtLogger;
            static TrtLogger &gLogger() { return uosm::utility::gTrtLogger; }

            // Core
            void shutdown();
            bool is_initialized() const
            {
                return (engine_ != nullptr) && (context_ != nullptr) && (stream_ != nullptr);
            }

            bool runInferenceFromGPU(const cv::cuda::GpuMat &gpu_img); // Zero-copy from GPU
            // Single image inference (batch=1)
            bool runInferenceSingleImage(cv::Mat &image);
            // Stereo inference (batch=2) - process left/right images in single TensorRT call
            bool runInferenceStereoFromGPU(const cv::cuda::GpuMat &gpu_img_left,
                                           const cv::cuda::GpuMat &gpu_img_right);

            // Helpers
            void copyOutputsToHostSingle();
            void parseKeypointsSingle(std::vector<cv::Point2f> &kpts, std::vector<float> &scores,
                                      std::vector<std::array<float, 4>> *cov_out = nullptr);
            void copyOutputsToHostStereo();
            void parseKeypointsStereo(StereoFeatureData &stereo_data);
            bool hasCovariance() const { return has_covariance_; }
            float getScale() const { return data_.scale_; }
            float getXOffset() const { return data_.x_offset_; }
            float getYOffset() const { return data_.y_offset_; }
            int getDescriptorDim() const { return data_.descriptor_dim_; }
            int getMaxKeypoints() const { return params_.max_keypoints; }
            int getBatchSize() const { return params_.batch_size; }
            void *getDescriptorsGpuBuffer() const;
            void *getKeypointsGpuBuffer() const;
            const float *getKeypointsGpuBufferBatch(int batch_idx) const;
            int getNumKeypointsFromGPU() const;
            std::pair<int, int> getNumKeypointsStereoFromGPU() const;
            cv::cuda::GpuMat wrapDescriptorsAsGpuMat(int num_keypoints) const;
            cv::cuda::GpuMat wrapDescriptorsAsGpuMatStereo(int batch_idx, int num_keypoints) const;

        private:
            FeatureExtractorParams params_;
            FeatureExtractionData data_;

            std::string input_tensor_name_;           // FLOAT32
            std::string output_keypoints_name_;       // FLOAT32
            std::string output_descriptors_name_;     // FLOAT32
            std::string output_num_keypoints_name_;   // INT64 / INT32
            std::string output_keypoint_scores_name_; // FLOAT32
            std::string output_covariances_name_;     // FLOAT32 (B,K,2,2); empty if absent
            bool has_covariance_ = false;
            int covariance_floats_ = 0; // floats per keypoint (4 = 2x2; 3 = packed)
            size_t num_kpts_element_size_;

            // TensorRT
            std::unique_ptr<nvinfer1::IRuntime> runtime_;
            std::unique_ptr<nvinfer1::ICudaEngine> engine_;
            std::unique_ptr<nvinfer1::IExecutionContext> context_;
            std::unordered_map<std::string, std::unique_ptr<uosm::utility::DynamicOutputAllocator>> mAllocatorMap;
            float *d_input_;
            cudaStream_t stream_;
            cudaEvent_t d2h_done_event_ = nullptr;

            void init();
            bool initTensorRT();
            void discoverTensorNames();
            void allocateBuffers();
            void setupBindingsAndAllocators();
            bool warmup(int n_iters = 3);
        };
    } // namespace perception
} // namespace uosm
#endif // FEATURE_EXTRACTOR_HPP_