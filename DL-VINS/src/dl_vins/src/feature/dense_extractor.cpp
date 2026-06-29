#include "../include/feature/dense_extractor.hpp"
#include "../include/feature/type_conv_helper.cuh"

#include <NvInferPlugin.h>
#include <rclcpp/logging.hpp>

#include <fstream>
#include <vector>

namespace uosm
{
    namespace perception
    {
        DenseExtractor::DenseExtractor(Params p) : params_(p)
        {
            if (!initTensorRT())
            {
                RCLCPP_ERROR(rclcpp::get_logger("dense_extractor"),
                             "Failed to initialize TensorRT for engine %s",
                             params_.engine_path.c_str());
                return;
            }
            discoverTensorNames();
            if (!allocateBuffers())
                return;
            setupBindings();
            warmup(3);
        }

        DenseExtractor::~DenseExtractor()
        {
            if (internal_stream_)
                cudaStreamSynchronize(internal_stream_);
            uosm::utility::resetTrtEngine(context_, engine_, runtime_);
            if (d_input_)
            {
                cudaFree(d_input_);
                d_input_ = nullptr;
            }
            if (d_score_map_)
            {
                cudaFree(d_score_map_);
                d_score_map_ = nullptr;
            }
            if (d_feature_map_)
            {
                cudaFree(d_feature_map_);
                d_feature_map_ = nullptr;
            }
            if (internal_stream_)
            {
                cudaStreamDestroy(internal_stream_);
                internal_stream_ = nullptr;
            }
        }

        bool DenseExtractor::initTensorRT()
        {
            initLibNvInferPlugins(&uosm::utility::gTrtLogger, "");

            engine_ = uosm::utility::loadCudaEngine(params_.engine_path, runtime_);
            if (!engine_)
            {
                RCLCPP_ERROR(rclcpp::get_logger("dense_extractor"),
                             "Failed to load engine: %s", params_.engine_path.c_str());
                return false;
            }
            context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
            if (!context_)
                return false;

            cudaError_t err = cudaStreamCreate(&internal_stream_);
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("dense_extractor"),
                             "cudaStreamCreate failed: %s", cudaGetErrorString(err));
                return false;
            }
            return true;
        }

        void DenseExtractor::discoverTensorNames()
        {
            int32_t n = engine_->getNbIOTensors();
            for (int32_t i = 0; i < n; ++i)
            {
                const char *name = engine_->getIOTensorName(i);
                std::string s(name);
                auto mode = engine_->getTensorIOMode(name);
                auto shape = engine_->getTensorShape(name);

                if (mode == nvinfer1::TensorIOMode::kINPUT)
                {
                    if (s == "image" || s == "images")
                    {
                        image_name_ = s;
                        if (shape.nbDims >= 4)
                        {
                            params_.batch_size = shape.d[0];
                            params_.input_channels = shape.d[1];
                            params_.input_h = shape.d[2];
                            params_.input_w = shape.d[3];
                        }
                    }
                }
                else
                {
                    if (s == "score_map")
                    {
                        score_map_name_ = s;
                    }
                    else if (s == "feature_map")
                    {
                        feature_map_name_ = s;
                        if (shape.nbDims >= 2)
                            params_.feature_dim = shape.d[1];
                    }
                }

                RCLCPP_INFO(rclcpp::get_logger("dense_extractor"),
                            "%s tensor: %s",
                            mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output", name);
            }
            RCLCPP_INFO(rclcpp::get_logger("dense_extractor"),
                        "DenseExtractor ready: B=%d %dx%d C=%d Cf=%d",
                        params_.batch_size, params_.input_h, params_.input_w,
                        params_.input_channels, params_.feature_dim);
        }

        bool DenseExtractor::allocateBuffers()
        {
            const size_t in_bytes = static_cast<size_t>(params_.batch_size) *
                                    params_.input_channels *
                                    params_.input_h * params_.input_w * sizeof(float);
            const size_t score_bytes = static_cast<size_t>(params_.batch_size) *
                                       params_.input_h * params_.input_w * sizeof(float);
            const size_t fmap_bytes = static_cast<size_t>(params_.batch_size) *
                                      params_.feature_dim *
                                      params_.input_h * params_.input_w * sizeof(float);

            cudaError_t e1 = cudaMalloc(reinterpret_cast<void **>(&d_input_), in_bytes);
            cudaError_t e2 = cudaMalloc(reinterpret_cast<void **>(&d_score_map_), score_bytes);
            cudaError_t e3 = cudaMalloc(reinterpret_cast<void **>(&d_feature_map_), fmap_bytes);
            if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("dense_extractor"),
                             "cudaMalloc failed: in=%s score=%s fmap=%s",
                             cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3));
                return false;
            }
            return true;
        }

        void DenseExtractor::setupBindings()
        {
            // Inputs and outputs are fixed-shape; bind once. Inputs may need
            // re-binding on every call if the input shape can change at runtime,
            // but for the static B=1 / B=2 export, shape is fixed and we can
            // call setInputShape once at warmup.
            context_->setInputShape(image_name_.c_str(),
                                    nvinfer1::Dims4{params_.batch_size,
                                                    params_.input_channels,
                                                    params_.input_h,
                                                    params_.input_w});
            if (!image_name_.empty())
                context_->setTensorAddress(image_name_.c_str(), d_input_);
            if (!score_map_name_.empty())
                context_->setTensorAddress(score_map_name_.c_str(), d_score_map_);
            if (!feature_map_name_.empty())
                context_->setTensorAddress(feature_map_name_.c_str(), d_feature_map_);
        }

        bool DenseExtractor::warmup(int n_iters)
        {
            if (!is_initialized())
                return false;
            for (int i = 0; i < n_iters; ++i)
            {
                if (!context_->enqueueV3(internal_stream_))
                    return false;
            }
            cudaStreamSynchronize(internal_stream_);
            RCLCPP_INFO(rclcpp::get_logger("dense_extractor"),
                        "Warmup complete (%d iters, B=%d %dx%d)",
                        n_iters, params_.batch_size, params_.input_h, params_.input_w);
            return true;
        }

        bool DenseExtractor::runInferenceStereoFromGPU(const cv::cuda::GpuMat &gpu_img_left,
                                                       const cv::cuda::GpuMat &gpu_img_right,
                                                       cudaStream_t stream)
        {
            if (!is_initialized() || !d_input_ || !d_score_map_ || !d_feature_map_)
                return false;
            if (params_.batch_size != 2)
            {
                RCLCPP_ERROR(rclcpp::get_logger("dense_extractor"),
                             "runInferenceStereoFromGPU requires batch_size=2");
                return false;
            }

            // Compute letterbox parameters (same for both — stereo is rectified)
            const float scale_w = static_cast<float>(params_.input_w) / gpu_img_left.cols;
            const float scale_h = static_cast<float>(params_.input_h) / gpu_img_left.rows;
            const float scale = std::min(scale_w, scale_h);
            const int new_w = static_cast<int>(gpu_img_left.cols * scale);
            const int new_h = static_cast<int>(gpu_img_left.rows * scale);
            const int pad_w = (params_.input_w - new_w) / 2;
            const int pad_h = (params_.input_h - new_h) / 2;
            scale_ = 1.0f / scale;
            x_offset_ = static_cast<float>(pad_w);
            y_offset_ = static_cast<float>(pad_h);

            // Single fused kernel: bilinear resize + letterbox pad + BGR->RGB (or
            // gray-broadcast for 1ch) + /255 + NCHW write, both batches in one launch.
            launchFusedPreprocess_RGB_Stereo(gpu_img_left, gpu_img_right, d_input_,
                                             params_.input_h, params_.input_w,
                                             new_w, new_h, pad_w, pad_h, stream);

            if (!context_->enqueueV3(stream))
            {
                RCLCPP_ERROR(rclcpp::get_logger("dense_extractor"),
                             "enqueueV3 failed");
                return false;
            }
            return true;
        }

        bool DenseExtractor::runInferenceFromGPU(const cv::cuda::GpuMat &gpu_img,
                                                 cudaStream_t stream)
        {
            if (!is_initialized() || !d_input_ || !d_score_map_ || !d_feature_map_)
                return false;
            if (params_.batch_size != 1)
            {
                RCLCPP_ERROR(rclcpp::get_logger("dense_extractor"),
                             "runInferenceFromGPU requires batch_size=1");
                return false;
            }

            const float scale_w = static_cast<float>(params_.input_w) / gpu_img.cols;
            const float scale_h = static_cast<float>(params_.input_h) / gpu_img.rows;
            const float scale = std::min(scale_w, scale_h);
            const int new_w = static_cast<int>(gpu_img.cols * scale);
            const int new_h = static_cast<int>(gpu_img.rows * scale);
            const int pad_w = (params_.input_w - new_w) / 2;
            const int pad_h = (params_.input_h - new_h) / 2;
            scale_ = 1.0f / scale;
            x_offset_ = static_cast<float>(pad_w);
            y_offset_ = static_cast<float>(pad_h);

            // Single fused kernel: bilinear resize + letterbox pad + BGR->RGB (or
            // gray-broadcast for 1ch) + /255 + NCHW write, all in one launch.
            launchFusedPreprocess_RGB_Single(gpu_img, d_input_,
                                             params_.input_h, params_.input_w,
                                             new_w, new_h, pad_w, pad_h, stream);

            if (!context_->enqueueV3(stream))
            {
                RCLCPP_ERROR(rclcpp::get_logger("dense_extractor"),
                             "enqueueV3 failed");
                return false;
            }
            return true;
        }
    } // namespace perception
} // namespace uosm
