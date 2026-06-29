#include "../include/feature/descriptor_head.hpp"

#include <NvInferPlugin.h>
#include <rclcpp/logging.hpp>

#include <fstream>
#include <vector>

namespace uosm
{
    namespace perception
    {
        DescriptorHead::DescriptorHead(Params p) : params_(p)
        {
            if (!initTensorRT())
            {
                RCLCPP_ERROR(rclcpp::get_logger("descriptor_head"),
                             "Failed to initialize TensorRT for engine %s",
                             params_.engine_path.c_str());
                return;
            }
            discoverTensorNames();
            allocateBuffers();
            setupBindings();
            warmup(3);
        }

        bool DescriptorHead::warmup(int n_iters)
        {
            if (!is_initialized() || !d_descriptors_)
                return false;
            if (feature_map_name_.empty() || kpts_name_.empty() || desc_name_.empty())
                return false;

            const size_t fmap_bytes = static_cast<size_t>(params_.batch) *
                                      params_.feature_dim *
                                      params_.model_h * params_.model_w * sizeof(float);
            const size_t kpts_bytes = static_cast<size_t>(params_.batch) *
                                      params_.max_kp * 2 * sizeof(float);

            void *d_fmap_tmp = nullptr;
            void *d_kpts_tmp = nullptr;
            if (cudaMalloc(&d_fmap_tmp, fmap_bytes) != cudaSuccess ||
                cudaMalloc(&d_kpts_tmp, kpts_bytes) != cudaSuccess)
            {
                if (d_fmap_tmp)
                    cudaFree(d_fmap_tmp);
                if (d_kpts_tmp)
                    cudaFree(d_kpts_tmp);
                RCLCPP_WARN(rclcpp::get_logger("descriptor_head"),
                            "Warmup: failed to allocate temp input buffers");
                return false;
            }

            std::vector<uosm::utility::WarmupBinding> bindings = {
                {feature_map_name_.c_str(),
                 nvinfer1::Dims4{params_.batch, params_.feature_dim, params_.model_h, params_.model_w},
                 d_fmap_tmp, fmap_bytes},
                {kpts_name_.c_str(),
                 nvinfer1::Dims3{params_.batch, params_.max_kp, 2},
                 d_kpts_tmp, kpts_bytes},
            };

            const bool ok = uosm::utility::warmupEngine(context_.get(), stream_, bindings, n_iters);

            cudaFree(d_fmap_tmp);
            cudaFree(d_kpts_tmp);

            if (!ok)
            {
                RCLCPP_WARN(rclcpp::get_logger("descriptor_head"), "Warmup failed");
                return false;
            }
            RCLCPP_INFO(rclcpp::get_logger("descriptor_head"),
                        "Warmup complete (%d iters, B=%d K=%d %dx%d D=%d)",
                        n_iters, params_.batch, params_.max_kp,
                        params_.model_h, params_.model_w, params_.feature_dim);
            return true;
        }

        DescriptorHead::~DescriptorHead()
        {
            if (stream_)
                cudaStreamSynchronize(stream_);
            uosm::utility::resetTrtEngine(context_, engine_, runtime_);
            if (d_descriptors_)
            {
                cudaFree(d_descriptors_);
                d_descriptors_ = nullptr;
            }
            if (stream_)
            {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
        }

        bool DescriptorHead::initTensorRT()
        {
            initLibNvInferPlugins(&uosm::utility::gTrtLogger, "");

            engine_ = uosm::utility::loadCudaEngine(params_.engine_path, runtime_);
            if (!engine_)
            {
                RCLCPP_ERROR(rclcpp::get_logger("descriptor_head"),
                             "Failed to load engine: %s", params_.engine_path.c_str());
                return false;
            }
            context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
            if (!context_)
                return false;

            cudaError_t err = cudaStreamCreate(&stream_);
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("descriptor_head"),
                             "cudaStreamCreate failed: %s", cudaGetErrorString(err));
                return false;
            }
            return true;
        }

        void DescriptorHead::discoverTensorNames()
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
                    if (s == "feature_map" || s == "fmap")
                    {
                        feature_map_name_ = s;
                        if (shape.nbDims >= 4)
                        {
                            params_.batch = shape.d[0];
                            params_.feature_dim = shape.d[1];
                            params_.model_h = shape.d[2];
                            params_.model_w = shape.d[3];
                        }
                    }
                    else if (s == "kpts_n" || s == "keypoints" || s == "kpts")
                    {
                        kpts_name_ = s;
                        if (shape.nbDims >= 2)
                            params_.max_kp = shape.d[1];
                    }
                }
                else
                {
                    if (s == "descriptors" || s == "descs")
                    {
                        desc_name_ = s;
                        if (shape.nbDims >= 3)
                            desc_dim_ = shape.d[2];
                    }
                }

                RCLCPP_INFO(rclcpp::get_logger("descriptor_head"),
                            "%s tensor: %s",
                            mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output",
                            name);
            }
            RCLCPP_INFO(rclcpp::get_logger("descriptor_head"),
                        "DescriptorHead ready: B=%d K=%d HxW=%dx%d Cf=%d D=%d",
                        params_.batch, params_.max_kp, params_.model_h, params_.model_w,
                        params_.feature_dim, desc_dim_);
        }

        void DescriptorHead::allocateBuffers()
        {
            descriptors_bytes_ = static_cast<size_t>(params_.batch) *
                                 static_cast<size_t>(params_.max_kp) *
                                 static_cast<size_t>(desc_dim_) * sizeof(float);
            cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_descriptors_), descriptors_bytes_);
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("descriptor_head"),
                             "cudaMalloc descriptors (%zu bytes) failed: %s",
                             descriptors_bytes_, cudaGetErrorString(err));
                d_descriptors_ = nullptr;
            }
        }

        void DescriptorHead::setupBindings()
        {
            if (!desc_name_.empty() && d_descriptors_)
                context_->setTensorAddress(desc_name_.c_str(), d_descriptors_);
        }

        bool DescriptorHead::runInference(const float *d_feature_map,
                                          const float *d_kpts_n,
                                          cudaStream_t external_stream)
        {
            if (!is_initialized() || !d_descriptors_)
                return false;
            if (feature_map_name_.empty() || kpts_name_.empty() || desc_name_.empty())
            {
                RCLCPP_ERROR(rclcpp::get_logger("descriptor_head"),
                             "tensor name discovery failed: fmap=%s kpts=%s desc=%s",
                             feature_map_name_.c_str(), kpts_name_.c_str(), desc_name_.c_str());
                return false;
            }

            context_->setTensorAddress(feature_map_name_.c_str(),
                                       const_cast<float *>(d_feature_map));
            context_->setTensorAddress(kpts_name_.c_str(),
                                       const_cast<float *>(d_kpts_n));
            context_->setTensorAddress(desc_name_.c_str(), d_descriptors_);

            cudaStream_t s = (external_stream != nullptr) ? external_stream : stream_;
            if (!context_->enqueueV3(s))
            {
                RCLCPP_ERROR(rclcpp::get_logger("descriptor_head"), "enqueueV3 failed");
                return false;
            }
            return true;
        }

        bool DescriptorHead::runInference(const float *d_feature_map, const float *d_kpts_n)
        {
            if (!runInference(d_feature_map, d_kpts_n, stream_))
                return false;
            cudaError_t sync_err = cudaStreamSynchronize(stream_);
            if (sync_err != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("descriptor_head"),
                             "stream sync failed: %s", cudaGetErrorString(sync_err));
                return false;
            }
            return true;
        }
    } // namespace perception
} // namespace uosm
