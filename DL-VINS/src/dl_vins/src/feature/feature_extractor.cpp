#include "../include/feature/feature_extractor.hpp"
#include <rclcpp/logging.hpp>

namespace uosm
{
    namespace perception
    {
        FeatureExtractor::FeatureExtractor(FeatureExtractorParams params) : params_(params)
        {
            init();
        }

        FeatureExtractor::~FeatureExtractor()
        {
            shutdown();
        }

        void FeatureExtractor::shutdown()
        {
            if (is_initialized() && stream_)
            {
                cudaStreamSynchronize(stream_);
            }

            mAllocatorMap.clear();

            uosm::utility::resetTrtEngine(context_, engine_, runtime_);

            if (d_input_)
            {
                cudaFree(d_input_);
                d_input_ = nullptr;
            }

            if (d2h_done_event_)
            {
                cudaEventDestroy(d2h_done_event_);
                d2h_done_event_ = nullptr;
            }

            if (stream_)
            {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }

            if (data_.h_pinned_buffer_)
            {
                cudaFreeHost(data_.h_pinned_buffer_);
                data_.h_pinned_buffer_ = nullptr;
            }
            RCLCPP_INFO(rclcpp::get_logger("feature_extractor"), "FeatureExtractor TensorRT & CUDA resources cleaned up.");
        }

        void FeatureExtractor::init()
        {
            if (!initTensorRT())
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "Failed to initialize TensorRT");
                return;
            }
            discoverTensorNames();
            allocateBuffers();
            setupBindingsAndAllocators();
            warmup(3);
        }

        // Run the full pipeline (preprocessing + TRT + post-process + D2H) on
        // a dummy GpuMat so every kernel along the runtime path pays its
        // first-call cost here instead of on frame 1.
        bool FeatureExtractor::warmup(int n_iters)
        {
            if (!is_initialized())
                return false;

            const int dummy_w = params_.input_width + 32;  // Add some padding to trigger any dynamic shape logic in the engine (if present)
            const int dummy_h = params_.input_height + 32; // Add some padding to trigger any dynamic shape logic in the engine (if present)
            const int cv_type = (params_.input_channels == 3) ? CV_8UC3 : CV_8UC1;
            cv::cuda::GpuMat dummy(dummy_h, dummy_w, cv_type);
            dummy.setTo(cv::Scalar::all(0));

            for (int i = 0; i < n_iters; ++i)
            {
                if (params_.batch_size == 2)
                {
                    if (!runInferenceStereoFromGPU(dummy, dummy))
                    {
                        RCLCPP_WARN(rclcpp::get_logger("feature_extractor"),
                                    "Warmup stereo iter %d failed", i);
                        return false;
                    }
                    copyOutputsToHostStereo();
                }
                else
                {
                    if (!runInferenceFromGPU(dummy))
                    {
                        RCLCPP_WARN(rclcpp::get_logger("feature_extractor"),
                                    "Warmup mono iter %d failed", i);
                        return false;
                    }
                    copyOutputsToHostSingle();
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            RCLCPP_INFO(rclcpp::get_logger("feature_extractor"),
                        "Pipeline warmup complete (%d iters, batch=%d, %dx%d)",
                        n_iters, params_.batch_size,
                        params_.input_height, params_.input_width);
            return true;
        }

        bool FeatureExtractor::initTensorRT()
        {
            bool plugins_loaded = initLibNvInferPlugins(&gLogger(), "");
            if (!plugins_loaded)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "Failed to initialize TensorRT plugins. Please build and link the TensorRT OSS");
                return false;
            }

            engine_ = uosm::utility::loadCudaEngine(params_.engine_path, runtime_, gLogger());
            if (!engine_)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "Failed to load engine at %s", params_.engine_path.c_str());
                return false;
            }

            context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
            if (!context_)
                return false;

            CUDA_CHECK(cudaStreamCreate(&stream_));
            CUDA_CHECK(cudaEventCreateWithFlags(&d2h_done_event_, cudaEventDisableTiming));
            return true;
        }

        void FeatureExtractor::discoverTensorNames()
        {
            int32_t nbIOTensors = engine_->getNbIOTensors();
            for (int32_t i = 0; i < nbIOTensors; ++i)
            {
                const char *tensorName = engine_->getIOTensorName(i);
                auto dtype = engine_->getTensorDataType(tensorName);
                std::string dtype_str = "OTHER";
                if (dtype == nvinfer1::DataType::kFLOAT)
                    dtype_str = "FLOAT32";
                else if (dtype == nvinfer1::DataType::kHALF)
                    dtype_str = "HALF";
                else if (dtype == nvinfer1::DataType::kINT64)
                    dtype_str = "INT64";
                else if (dtype == nvinfer1::DataType::kINT32)
                    dtype_str = "INT32";
                else if (dtype == nvinfer1::DataType::kINT8)
                    dtype_str = "INT8";
                else if (dtype == nvinfer1::DataType::kBOOL)
                    dtype_str = "BOOL";

                if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
                {
                    input_tensor_name_ = tensorName;

                    // Query engine's actual input dimensions (override config if different)
                    auto input_shape = engine_->getTensorShape(tensorName);
                    if (input_shape.nbDims >= 4)
                    {
                        int engine_channels = input_shape.d[1];
                        int engine_height = input_shape.d[2];
                        int engine_width = input_shape.d[3];

                        if (engine_width != params_.input_width || engine_height != params_.input_height)
                        {
                            RCLCPP_INFO(rclcpp::get_logger("feature_extractor"),
                                        "Engine input dimensions [%dx%d] differ from config [%dx%d], using engine dimensions",
                                        engine_height, engine_width, params_.input_height, params_.input_width);
                            params_.input_width = engine_width;
                            params_.input_height = engine_height;
                        }
                        if (engine_channels != params_.input_channels)
                        {
                            RCLCPP_INFO(rclcpp::get_logger("feature_extractor"),
                                        "Engine channels [%d] differ from config [%d], using engine channels",
                                        engine_channels, params_.input_channels);
                            params_.input_channels = engine_channels;
                        }
                    }
                    RCLCPP_INFO(rclcpp::get_logger("feature_extractor"),
                                "Found Input tensor: %s with dtype: %s, shape: [%dx%d]",
                                tensorName, dtype_str.c_str(), params_.input_height, params_.input_width);
                }
                else
                {
                    std::string name(tensorName);
                    if (name == "keypoints")
                        output_keypoints_name_ = name;
                    else if (name == "keypoint_scores")
                        output_keypoint_scores_name_ = name;
                    else if (name == "descriptors")
                    {
                        output_descriptors_name_ = name;
                        auto desc_shape = engine_->getTensorShape(tensorName);
                        if (desc_shape.nbDims >= 3)
                        {
                            data_.descriptor_dim_ = desc_shape.d[2];
                        }
                        else
                        {
                            data_.descriptor_dim_ = 256;
                        }
                        RCLCPP_INFO(rclcpp::get_logger("feature_extractor"), "Descriptor dimension: %d", data_.descriptor_dim_);
                    }
                    else if (name == "num_keypoints")
                        output_num_keypoints_name_ = name;
                    else if (name == "covariances")
                    {
                        output_covariances_name_ = name;
                        has_covariance_ = true;
                        auto cov_shape = engine_->getTensorShape(tensorName);
                        // (B,K,2,2) -> 4 floats/kpt; (B,K,3) packed -> 3 floats/kpt.
                        if (cov_shape.nbDims >= 4)
                            covariance_floats_ = cov_shape.d[cov_shape.nbDims - 1] *
                                                 cov_shape.d[cov_shape.nbDims - 2];
                        else if (cov_shape.nbDims == 3)
                            covariance_floats_ = cov_shape.d[2];
                        else
                            covariance_floats_ = 4;
                        RCLCPP_INFO(rclcpp::get_logger("feature_extractor"),
                                    "Covariance head detected: %d floats/keypoint", covariance_floats_);
                    }
                    RCLCPP_INFO(rclcpp::get_logger("feature_extractor"), "Found Output tensor: %s with dtype: %s", tensorName, dtype_str.c_str());
                }
            }
            RCLCPP_INFO(rclcpp::get_logger("feature_extractor"),
                        "Using I/O tensors: Input='%s', Keypoints='%s', Scores='%s', Descriptors='%s', NumKeypoints='%s'",
                        input_tensor_name_.c_str(), output_keypoints_name_.c_str(),
                        output_keypoint_scores_name_.c_str(), output_descriptors_name_.c_str(),
                        output_num_keypoints_name_.c_str());
        }

        void FeatureExtractor::allocateBuffers()
        {
            size_t num_kpts_element_size = 0;
            if (!output_num_keypoints_name_.empty())
            {
                auto num_kpts_dtype = engine_->getTensorDataType(output_num_keypoints_name_.c_str());
                if (num_kpts_dtype == nvinfer1::DataType::kINT32)
                {
                    num_kpts_element_size = sizeof(int32_t);
                }
                else if (num_kpts_dtype == nvinfer1::DataType::kINT64)
                {
                    num_kpts_element_size = sizeof(int64_t);
                }
                else
                {
                    RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "Unsupported num_keypoints data type");
                    return;
                }
            }

            // Support batch_size=1 (monocular) or batch_size=2 (stereo)
            const int batch = params_.batch_size;
            size_t num_kpts_raw_size = num_kpts_element_size * batch;
            size_t keypoints_raw_size = params_.max_keypoints * 2 * sizeof(float) * batch;
            size_t scores_raw_size = params_.max_keypoints * sizeof(float) * batch;
            size_t descriptors_raw_size = params_.max_keypoints * data_.descriptor_dim_ * sizeof(float) * batch;

            constexpr size_t ALIGNMENT = 256U;
            auto alignUp = [](size_t size, size_t alignment)
            {
                return (size + alignment - 1) & ~(alignment - 1);
            };

            size_t covariances_raw_size = has_covariance_
                                              ? params_.max_keypoints * covariance_floats_ * sizeof(float) * batch
                                              : 0U;

            size_t num_kpts_size = alignUp(num_kpts_raw_size, ALIGNMENT);
            size_t keypoints_size = alignUp(keypoints_raw_size, ALIGNMENT);
            size_t scores_size = alignUp(scores_raw_size, ALIGNMENT);
            size_t descriptors_size = alignUp(descriptors_raw_size, ALIGNMENT);
            size_t covariances_size = alignUp(covariances_raw_size, ALIGNMENT);

            data_.total_output_size_ = num_kpts_size + keypoints_size + scores_size + descriptors_size + covariances_size;
            num_kpts_element_size_ = num_kpts_element_size;

            CUDA_CHECK(cudaMallocHost(&data_.h_pinned_buffer_, data_.total_output_size_));

            data_.num_kpts_offset_ = 0;
            data_.keypoints_offset_ = data_.num_kpts_offset_ + num_kpts_size;
            data_.scores_offset_ = data_.keypoints_offset_ + keypoints_size;
            data_.descriptors_offset_ = data_.scores_offset_ + scores_size;
            data_.covariances_offset_ = data_.descriptors_offset_ + descriptors_size;

            // Input size: batch * channels * height * width
            size_t input_size = batch * params_.input_channels * params_.input_height * params_.input_width * sizeof(float);

            if (params_.use_unified_memory)
            {
                CUDA_CHECK(cudaMallocManaged(&d_input_, input_size));
#if CUDART_VERSION >= 13000
                cudaMemLocation gpu_loc;
                gpu_loc.type = cudaMemLocationTypeDevice;
                gpu_loc.id = 0;
                CUDA_CHECK(cudaMemAdvise(d_input_, input_size, cudaMemAdviseSetPreferredLocation, gpu_loc));

                cudaMemLocation cpu_loc;
                cpu_loc.type = cudaMemLocationTypeHost;
                cpu_loc.id = 0;
                CUDA_CHECK(cudaMemAdvise(d_input_, input_size, cudaMemAdviseSetAccessedBy, cpu_loc));
#else
                CUDA_CHECK(cudaMemAdvise(d_input_, input_size, cudaMemAdviseSetPreferredLocation, 0));
                CUDA_CHECK(cudaMemAdvise(d_input_, input_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
#endif
            }
            else
            {
                CUDA_CHECK(cudaMalloc(&d_input_, input_size));
            }
            // Pre-zero so the fused preprocess kernel can skip writing the letterbox border.
            CUDA_CHECK(cudaMemset(d_input_, 0, input_size));
            RCLCPP_INFO(rclcpp::get_logger("feature_extractor"),
                        "Allocated %.2f MB pinned memory for %s model (%dCH), descriptor_dim: %d",
                        data_.total_output_size_ / (1024.0 * 1024.0), params_.model_type.c_str(),
                        params_.input_channels, data_.descriptor_dim_);
        }

        void FeatureExtractor::setupBindingsAndAllocators()
        {
            // auto print_shape = [this](const char *name)
            // {
            //     if (std::string(name).empty())
            //         return;
            //     auto shape = engine_->getTensorShape(name);
            //     std::string shape_str;
            //     for (int i = 0; i < shape.nbDims; ++i)
            //         shape_str += std::to_string(shape.d[i]) + ", ";
            //     RCLCPP_INFO(rclcpp::get_logger("feature_extractor"), "Tensor '%s' shape: [ %s]", name, shape_str.c_str());
            // };
            // print_shape(input_tensor_name_.c_str());
            // print_shape(output_keypoints_name_.c_str());
            // print_shape(output_keypoint_scores_name_.c_str());
            // print_shape(output_descriptors_name_.c_str());
            // print_shape(output_num_keypoints_name_.c_str());

            context_->setInputTensorAddress(input_tensor_name_.c_str(), d_input_);
            context_->setInputShape(input_tensor_name_.c_str(),
                                    nvinfer1::Dims4{params_.batch_size,
                                                    params_.input_channels,
                                                    params_.input_height,
                                                    params_.input_width});

            for (const auto &name : {output_keypoints_name_, output_keypoint_scores_name_,
                                     output_descriptors_name_, output_num_keypoints_name_,
                                     output_covariances_name_})
            {
                if (!name.empty())
                {
                    auto allocator = std::make_unique<uosm::utility::DynamicOutputAllocator>();
                    context_->setOutputAllocator(name.c_str(), allocator.get());
                    mAllocatorMap.emplace(name, std::move(allocator));
                }
            }
        }

        bool FeatureExtractor::runInferenceSingleImage(cv::Mat &image)
        {
            // Compute letterbox parameters (maintains aspect ratio)
            const float scale_w = static_cast<float>(params_.input_width) / image.cols;
            const float scale_h = static_cast<float>(params_.input_height) / image.rows;
            const float scale = std::min(scale_w, scale_h); // Use smaller scale to fit

            const int new_w = static_cast<int>(image.cols * scale);
            const int new_h = static_cast<int>(image.rows * scale);
            const int pad_w = (params_.input_width - new_w) / 2;  // Left padding
            const int pad_h = (params_.input_height - new_h) / 2; // Top padding

            // Store inverse transform: scale from model coords to original coords
            data_.scale_ = 1.0f / scale;
            data_.x_offset_ = static_cast<float>(pad_w);
            data_.y_offset_ = static_cast<float>(pad_h);

            if (params_.profile_inference)
                nvtxRangePushA("Preprocessing_Single");
            {
                cv::cuda::GpuMat d_img_raw(image);
                if (params_.input_channels == 3)
                    launchFusedPreprocess_RGB_Single(d_img_raw, static_cast<float *>(d_input_),
                                                     params_.input_height, params_.input_width,
                                                     new_w, new_h, pad_w, pad_h, stream_);
                else
                    launchFusedPreprocess_Gray_Single(d_img_raw, static_cast<float *>(d_input_),
                                                      params_.input_height, params_.input_width,
                                                      new_w, new_h, pad_w, pad_h, stream_);
            }
            if (params_.profile_inference)
                nvtxRangePop(); // Preprocessing_Single

            // Set input shape for batch=1
            context_->setInputShape(input_tensor_name_.c_str(),
                                    nvinfer1::Dims4{1, params_.input_channels, params_.input_height, params_.input_width});

            if (params_.profile_inference)
                nvtxRangePushA("Inference_Single");
            if (!context_->enqueueV3(stream_))
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "Failed to enqueue single-image inference");
                return false;
            }
            if (params_.profile_inference)
                nvtxRangePop(); // Inference_Single

            // Post-process keypoints on GPU (transform from model coords to original image coords)
            if (params_.profile_inference)
                nvtxRangePushA("Postprocessing_Single");
            auto &keypoints_allocator = mAllocatorMap.at(output_keypoints_name_);
            void *num_kpts_buf = nullptr;
            if (!output_num_keypoints_name_.empty())
                num_kpts_buf = mAllocatorMap.at(output_num_keypoints_name_)->getBuffer();
            launchTransformKeypointsSingle(
                keypoints_allocator->getBuffer(),
                num_kpts_buf,
                num_kpts_element_size_,
                data_.scale_, data_.x_offset_, data_.y_offset_,
                params_.max_keypoints, params_.input_width, params_.input_height,
                (params_.model_type == "aliked"),
                stream_);
            if (params_.profile_inference)
                nvtxRangePop(); // Postprocessing_Single

            return true;
        }

        bool FeatureExtractor::runInferenceFromGPU(const cv::cuda::GpuMat &gpu_img)
        {
            // Compute letterbox parameters (maintains aspect ratio)
            const float scale_w = static_cast<float>(params_.input_width) / gpu_img.cols;
            const float scale_h = static_cast<float>(params_.input_height) / gpu_img.rows;
            const float scale = std::min(scale_w, scale_h);

            const int new_w = static_cast<int>(gpu_img.cols * scale);
            const int new_h = static_cast<int>(gpu_img.rows * scale);
            const int pad_w = (params_.input_width - new_w) / 2;
            const int pad_h = (params_.input_height - new_h) / 2;

            data_.scale_ = 1.0f / scale;
            data_.x_offset_ = static_cast<float>(pad_w);
            data_.y_offset_ = static_cast<float>(pad_h);

            if (params_.profile_inference)
                nvtxRangePushA("Preprocessing_GPU");
            if (params_.input_channels == 3)
                launchFusedPreprocess_RGB_Single(gpu_img, static_cast<float *>(d_input_),
                                                 params_.input_height, params_.input_width,
                                                 new_w, new_h, pad_w, pad_h, stream_);
            else
                launchFusedPreprocess_Gray_Single(gpu_img, static_cast<float *>(d_input_),
                                                  params_.input_height, params_.input_width,
                                                  new_w, new_h, pad_w, pad_h, stream_);
            if (params_.profile_inference)
                nvtxRangePop(); // Preprocessing_GPU

            // Set input shape for batch=1
            context_->setInputShape(input_tensor_name_.c_str(),
                                    nvinfer1::Dims4{1, params_.input_channels, params_.input_height, params_.input_width});

            if (params_.profile_inference)
                nvtxRangePushA("Inference_GPU");
            if (!context_->enqueueV3(stream_))
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "Failed to enqueue GPU inference");
                return false;
            }
            if (params_.profile_inference)
                nvtxRangePop(); // Inference_GPU

            // Post-process keypoints on GPU
            if (params_.profile_inference)
                nvtxRangePushA("Postprocessing_GPU");
            auto &keypoints_allocator = mAllocatorMap.at(output_keypoints_name_);
            void *num_kpts_buf = nullptr;
            if (!output_num_keypoints_name_.empty())
                num_kpts_buf = mAllocatorMap.at(output_num_keypoints_name_)->getBuffer();
            launchTransformKeypointsSingle(
                keypoints_allocator->getBuffer(),
                num_kpts_buf,
                num_kpts_element_size_,
                data_.scale_, data_.x_offset_, data_.y_offset_,
                params_.max_keypoints, params_.input_width, params_.input_height,
                (params_.model_type == "aliked"),
                stream_);
            if (params_.profile_inference)
                nvtxRangePop(); // Postprocessing_GPU

            return true;
        }

        void FeatureExtractor::copyOutputsToHostSingle()
        {
            const bool has_num_kpts = !output_num_keypoints_name_.empty();
            auto &keypoints_allocator = mAllocatorMap.at(output_keypoints_name_);
            auto &scores_allocator = mAllocatorMap.at(output_keypoint_scores_name_);

            void *num_kpts_buf = has_num_kpts ? mAllocatorMap.at(output_num_keypoints_name_)->getBuffer() : nullptr;
            void *keypoints_buf = keypoints_allocator->getBuffer();
            void *scores_buf = scores_allocator->getBuffer();

            if ((has_num_kpts && !num_kpts_buf) || !keypoints_buf || !scores_buf || !data_.h_pinned_buffer_)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "One or more buffers are null (single)");
                return;
            }

            // For batch=1, only copy single image's data
            size_t keypoints_size = params_.max_keypoints * 2 * sizeof(float);
            size_t scores_size = params_.max_keypoints * sizeof(float);

            if (has_num_kpts)
            {
                cudaError_t err = cudaMemcpyAsync(
                    static_cast<char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_,
                    num_kpts_buf,
                    num_kpts_element_size_,
                    cudaMemcpyDeviceToHost, stream_);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "CUDA error in num_keypoints copy (single): %s", cudaGetErrorString(err));
                    return;
                }
            }
            cudaError_t err = cudaSuccess;

            err = cudaMemcpyAsync(
                static_cast<char *>(data_.h_pinned_buffer_) + data_.keypoints_offset_,
                keypoints_buf,
                keypoints_size,
                cudaMemcpyDeviceToHost, stream_);
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "CUDA error in keypoints copy (single): %s", cudaGetErrorString(err));
                return;
            }

            err = cudaMemcpyAsync(
                static_cast<char *>(data_.h_pinned_buffer_) + data_.scores_offset_,
                scores_buf,
                scores_size,
                cudaMemcpyDeviceToHost, stream_);
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "CUDA error in scores copy (single): %s", cudaGetErrorString(err));
                return;
            }

            if (has_covariance_)
            {
                void *cov_buf = mAllocatorMap.at(output_covariances_name_)->getBuffer();
                if (cov_buf)
                {
                    size_t cov_size = params_.max_keypoints * covariance_floats_ * sizeof(float);
                    err = cudaMemcpyAsync(
                        static_cast<char *>(data_.h_pinned_buffer_) + data_.covariances_offset_,
                        cov_buf, cov_size, cudaMemcpyDeviceToHost, stream_);
                    if (err != cudaSuccess)
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "CUDA error in covariances copy (single): %s", cudaGetErrorString(err));
                        return;
                    }
                }
            }

            CUDA_CHECK(cudaEventRecord(d2h_done_event_, stream_));
        }

        void FeatureExtractor::parseKeypointsSingle(std::vector<cv::Point2f> &kpts, std::vector<float> &scores,
                                                    std::vector<std::array<float, 4>> *cov_out)
        {
            CUDA_CHECK(cudaEventSynchronize(d2h_done_event_));

            float *keypoints_ptr = reinterpret_cast<float *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.keypoints_offset_);
            float *scores_ptr = reinterpret_cast<float *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.scores_offset_);

            int64_t num_kpts;
            if (output_num_keypoints_name_.empty())
            {
                num_kpts = 0;
                for (int i = 0; i < params_.max_keypoints; ++i)
                {
                    if (scores_ptr[i] > 0.0f)
                        ++num_kpts;
                }
            }
            else if (num_kpts_element_size_ == sizeof(int32_t))
            {
                auto *num_kpts_ptr = reinterpret_cast<int32_t *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_);
                num_kpts = num_kpts_ptr[0];
            }
            else
            {
                auto *num_kpts_ptr = reinterpret_cast<int64_t *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_);
                num_kpts = num_kpts_ptr[0];
            }

            num_kpts = std::min(num_kpts, static_cast<int64_t>(params_.max_keypoints));

            kpts.resize(num_kpts);
            scores.resize(num_kpts);

            for (int i = 0; i < num_kpts; ++i)
            {
                kpts[i] = cv::Point2f(keypoints_ptr[i * 2], keypoints_ptr[i * 2 + 1]);
                scores[i] = scores_ptr[i];
            }

            if (has_covariance_ && cov_out)
            {
                const float *cov_ptr = reinterpret_cast<float *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.covariances_offset_);
                cov_out->resize(num_kpts);
                for (int i = 0; i < num_kpts; ++i)
                {
                    const float *c = cov_ptr + i * covariance_floats_;
                    // 4 floats: full 2x2 [c00,c01,c10,c11]; 3 floats: packed [s_uu,s_uv,s_vv].
                    (*cov_out)[i] = (covariance_floats_ >= 4)
                                        ? std::array<float, 4>{c[0], c[1], c[2], c[3]}
                                        : std::array<float, 4>{c[0], c[1], c[1], c[2]};
                }
            }
        }

        void *FeatureExtractor::getDescriptorsGpuBuffer() const
        {
            if (output_descriptors_name_.empty())
            {
                return nullptr;
            }
            auto it = mAllocatorMap.find(output_descriptors_name_);
            if (it == mAllocatorMap.end())
            {
                return nullptr;
            }
            return it->second->getBuffer();
        }

        void *FeatureExtractor::getKeypointsGpuBuffer() const
        {
            if (output_keypoints_name_.empty())
            {
                return nullptr;
            }
            auto it = mAllocatorMap.find(output_keypoints_name_);
            if (it == mAllocatorMap.end())
            {
                return nullptr;
            }
            return it->second->getBuffer();
        }

        const float *FeatureExtractor::getKeypointsGpuBufferBatch(int batch_idx) const
        {
            if (batch_idx < 0 || batch_idx >= params_.batch_size)
                return nullptr;
            void *buf = getKeypointsGpuBuffer();
            if (!buf)
                return nullptr;
            const size_t batch_stride = static_cast<size_t>(params_.max_keypoints) * 2;
            return static_cast<const float *>(buf) + batch_idx * batch_stride;
        }

        int FeatureExtractor::getNumKeypointsFromGPU() const
        {
            if (!data_.h_pinned_buffer_)
                return 0;

            if (output_num_keypoints_name_.empty())
            {
                const float *scores_ptr = reinterpret_cast<const float *>(
                    static_cast<const char *>(data_.h_pinned_buffer_) + data_.scores_offset_);
                int count = 0;
                for (int i = 0; i < params_.max_keypoints; ++i)
                    if (scores_ptr[i] > 0.0f)
                        ++count;
                return count;
            }

            // num_kpts already mirrored to the pinned buffer by copyOutputsToHostSingle.
            int64_t num_kpts;
            if (num_kpts_element_size_ == sizeof(int32_t))
            {
                auto *num_kpts_ptr = reinterpret_cast<const int32_t *>(
                    static_cast<const char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_);
                num_kpts = num_kpts_ptr[0];
            }
            else
            {
                auto *num_kpts_ptr = reinterpret_cast<const int64_t *>(
                    static_cast<const char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_);
                num_kpts = num_kpts_ptr[0];
            }

            return std::min(static_cast<int>(num_kpts), params_.max_keypoints);
        }

        cv::cuda::GpuMat FeatureExtractor::wrapDescriptorsAsGpuMat(int num_keypoints) const
        {
            void *desc_buf = getDescriptorsGpuBuffer();
            if (!desc_buf || num_keypoints <= 0)
            {
                return cv::cuda::GpuMat();
            }

            // Wrap existing GPU memory as GpuMat (no copy)
            // TensorRT output is row-major: (N, descriptor_dim) with float32
            return cv::cuda::GpuMat(
                num_keypoints,                        // rows = number of keypoints
                data_.descriptor_dim_,                // cols = descriptor dimension
                CV_32F,                               // float32
                desc_buf,                             // GPU pointer
                data_.descriptor_dim_ * sizeof(float) // step (bytes per row)
            );
        }

        bool FeatureExtractor::runInferenceStereoFromGPU(const cv::cuda::GpuMat &gpu_img_left,
                                                         const cv::cuda::GpuMat &gpu_img_right)
        {
            if (params_.batch_size != 2)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "runInferenceStereoFromGPU requires batch_size=2 model");
                return false;
            }

            // Compute letterbox parameters (same for both images - stereo is rectified)
            const float scale_w = static_cast<float>(params_.input_width) / gpu_img_left.cols;
            const float scale_h = static_cast<float>(params_.input_height) / gpu_img_left.rows;
            const float scale = std::min(scale_w, scale_h);

            const int new_w = static_cast<int>(gpu_img_left.cols * scale);
            const int new_h = static_cast<int>(gpu_img_left.rows * scale);
            const int pad_w = (params_.input_width - new_w) / 2;
            const int pad_h = (params_.input_height - new_h) / 2;

            data_.scale_ = 1.0f / scale;
            data_.x_offset_ = static_cast<float>(pad_w);
            data_.y_offset_ = static_cast<float>(pad_h);

            if (params_.profile_inference)
                nvtxRangePushA("Preprocessing_Stereo");
            if (params_.input_channels == 3)
                launchFusedPreprocess_RGB_Stereo(gpu_img_left, gpu_img_right,
                                                 static_cast<float *>(d_input_),
                                                 params_.input_height, params_.input_width,
                                                 new_w, new_h, pad_w, pad_h, stream_);
            else
                launchFusedPreprocess_Gray_Stereo(gpu_img_left, gpu_img_right,
                                                  static_cast<float *>(d_input_),
                                                  params_.input_height, params_.input_width,
                                                  new_w, new_h, pad_w, pad_h, stream_);
            if (params_.profile_inference)
                nvtxRangePop(); // Preprocessing_Stereo

            // Set input shape for batch=2
            context_->setInputShape(input_tensor_name_.c_str(),
                                    nvinfer1::Dims4{2, params_.input_channels, params_.input_height, params_.input_width});

            if (params_.profile_inference)
                nvtxRangePushA("Inference_Stereo");
            if (!context_->enqueueV3(stream_))
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "Failed to enqueue stereo inference");
                return false;
            }
            if (params_.profile_inference)
                nvtxRangePop(); // Inference_Stereo

            // Post-process keypoints on GPU for both images
            if (params_.profile_inference)
                nvtxRangePushA("Postprocessing_Stereo");
            auto &keypoints_allocator = mAllocatorMap.at(output_keypoints_name_);
            void *num_kpts_buf = nullptr;
            if (!output_num_keypoints_name_.empty())
                num_kpts_buf = mAllocatorMap.at(output_num_keypoints_name_)->getBuffer();
            launchTransformKeypointsStereo(
                keypoints_allocator->getBuffer(),
                num_kpts_buf,
                num_kpts_element_size_,
                data_.scale_, data_.x_offset_, data_.y_offset_,
                params_.max_keypoints, params_.input_width, params_.input_height,
                (params_.model_type == "aliked"),
                stream_);
            if (params_.profile_inference)
                nvtxRangePop(); // Postprocessing_Stereo

            return true;
        }

        void FeatureExtractor::copyOutputsToHostStereo()
        {
            const bool has_num_kpts = !output_num_keypoints_name_.empty();
            auto &keypoints_allocator = mAllocatorMap.at(output_keypoints_name_);
            auto &scores_allocator = mAllocatorMap.at(output_keypoint_scores_name_);

            void *num_kpts_buf = has_num_kpts ? mAllocatorMap.at(output_num_keypoints_name_)->getBuffer() : nullptr;
            void *keypoints_buf = keypoints_allocator->getBuffer();
            void *scores_buf = scores_allocator->getBuffer();

            if ((has_num_kpts && !num_kpts_buf) || !keypoints_buf || !scores_buf || !data_.h_pinned_buffer_)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "One or more buffers are null (stereo)");
                return;
            }

            // For batch=2, copy both images' data
            size_t keypoints_size = params_.max_keypoints * 2 * sizeof(float) * 2;
            size_t scores_size = params_.max_keypoints * sizeof(float) * 2;

            cudaError_t err = cudaSuccess;
            if (has_num_kpts)
            {
                err = cudaMemcpyAsync(
                    static_cast<char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_,
                    num_kpts_buf,
                    num_kpts_element_size_ * 2,
                    cudaMemcpyDeviceToHost, stream_);
                if (err != cudaSuccess)
                {
                    RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "CUDA error in num_keypoints copy (stereo): %s", cudaGetErrorString(err));
                    return;
                }
            }

            err = cudaMemcpyAsync(
                static_cast<char *>(data_.h_pinned_buffer_) + data_.keypoints_offset_,
                keypoints_buf,
                keypoints_size,
                cudaMemcpyDeviceToHost, stream_);
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "CUDA error in keypoints copy (stereo): %s", cudaGetErrorString(err));
                return;
            }

            err = cudaMemcpyAsync(
                static_cast<char *>(data_.h_pinned_buffer_) + data_.scores_offset_,
                scores_buf,
                scores_size,
                cudaMemcpyDeviceToHost, stream_);
            if (err != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "CUDA error in scores copy (stereo): %s", cudaGetErrorString(err));
                return;
            }

            if (has_covariance_)
            {
                void *cov_buf = mAllocatorMap.at(output_covariances_name_)->getBuffer();
                if (cov_buf)
                {
                    size_t cov_size = params_.max_keypoints * covariance_floats_ * sizeof(float) * 2;
                    err = cudaMemcpyAsync(
                        static_cast<char *>(data_.h_pinned_buffer_) + data_.covariances_offset_,
                        cov_buf, cov_size, cudaMemcpyDeviceToHost, stream_);
                    if (err != cudaSuccess)
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("feature_extractor"), "CUDA error in covariances copy (stereo): %s", cudaGetErrorString(err));
                        return;
                    }
                }
            }

            CUDA_CHECK(cudaEventRecord(d2h_done_event_, stream_));
        }

        void FeatureExtractor::parseKeypointsStereo(StereoFeatureData &stereo_data)
        {
            CUDA_CHECK(cudaEventSynchronize(d2h_done_event_));

            float *scores_ptr_for_count = reinterpret_cast<float *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.scores_offset_);
            int64_t num_kpts_left, num_kpts_right;

            if (output_num_keypoints_name_.empty())
            {
                num_kpts_left = 0;
                num_kpts_right = 0;
                for (int i = 0; i < params_.max_keypoints; ++i)
                {
                    if (scores_ptr_for_count[i] > 0.0f)
                        ++num_kpts_left;
                    if (scores_ptr_for_count[params_.max_keypoints + i] > 0.0f)
                        ++num_kpts_right;
                }
            }
            else if (num_kpts_element_size_ == sizeof(int32_t))
            {
                auto *num_kpts_ptr = reinterpret_cast<int32_t *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_);
                num_kpts_left = num_kpts_ptr[0];
                num_kpts_right = num_kpts_ptr[1];
            }
            else
            {
                auto *num_kpts_ptr = reinterpret_cast<int64_t *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_);
                num_kpts_left = num_kpts_ptr[0];
                num_kpts_right = num_kpts_ptr[1];
            }

            num_kpts_left = std::min(num_kpts_left, static_cast<int64_t>(params_.max_keypoints));
            num_kpts_right = std::min(num_kpts_right, static_cast<int64_t>(params_.max_keypoints));

            stereo_data.num_keypoints_left = static_cast<int>(num_kpts_left);
            stereo_data.num_keypoints_right = static_cast<int>(num_kpts_right);

            // Keypoints are stored as (2, max_keypoints, 2)
            float *keypoints_ptr = reinterpret_cast<float *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.keypoints_offset_);
            float *scores_ptr = reinterpret_cast<float *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.scores_offset_);

            // Parse left image keypoints (batch 0)
            stereo_data.keypoints_left.resize(num_kpts_left);
            stereo_data.scores_left.resize(num_kpts_left);
            for (int i = 0; i < num_kpts_left; ++i)
            {
                stereo_data.keypoints_left[i] = cv::Point2f(keypoints_ptr[i * 2], keypoints_ptr[i * 2 + 1]);
                stereo_data.scores_left[i] = scores_ptr[i];
            }

            // Parse right image keypoints (batch 1)
            // Offset by max_keypoints * 2 for keypoints, max_keypoints for scores
            float *keypoints_ptr_right = keypoints_ptr + params_.max_keypoints * 2;
            float *scores_ptr_right = scores_ptr + params_.max_keypoints;

            stereo_data.keypoints_right.resize(num_kpts_right);
            stereo_data.scores_right.resize(num_kpts_right);
            for (int i = 0; i < num_kpts_right; ++i)
            {
                stereo_data.keypoints_right[i] = cv::Point2f(keypoints_ptr_right[i * 2], keypoints_ptr_right[i * 2 + 1]);
                stereo_data.scores_right[i] = scores_ptr_right[i];
            }

            if (has_covariance_)
            {
                // Covariance buffer layout mirrors keypoints: (2, max_keypoints, covariance_floats_).
                const float *cov_ptr = reinterpret_cast<float *>(static_cast<char *>(data_.h_pinned_buffer_) + data_.covariances_offset_);
                const float *cov_ptr_right = cov_ptr + params_.max_keypoints * covariance_floats_;
                auto pack = [this](const float *c) -> std::array<float, 4>
                {
                    return (covariance_floats_ >= 4)
                               ? std::array<float, 4>{c[0], c[1], c[2], c[3]}
                               : std::array<float, 4>{c[0], c[1], c[1], c[2]};
                };
                stereo_data.cov_left.resize(num_kpts_left);
                for (int i = 0; i < num_kpts_left; ++i)
                    stereo_data.cov_left[i] = pack(cov_ptr + i * covariance_floats_);
                stereo_data.cov_right.resize(num_kpts_right);
                for (int i = 0; i < num_kpts_right; ++i)
                    stereo_data.cov_right[i] = pack(cov_ptr_right + i * covariance_floats_);
            }
        }

        std::pair<int, int> FeatureExtractor::getNumKeypointsStereoFromGPU() const
        {
            if (params_.batch_size != 2 || !data_.h_pinned_buffer_)
            {
                return {0, 0};
            }

            if (output_num_keypoints_name_.empty())
            {
                const float *scores_ptr = reinterpret_cast<const float *>(
                    static_cast<const char *>(data_.h_pinned_buffer_) + data_.scores_offset_);
                int n_left = 0, n_right = 0;
                for (int i = 0; i < params_.max_keypoints; ++i)
                {
                    if (scores_ptr[i] > 0.0f)
                        ++n_left;
                    if (scores_ptr[params_.max_keypoints + i] > 0.0f)
                        ++n_right;
                }
                return {n_left, n_right};
            }

            // num_kpts already mirrored to the pinned buffer by copyOutputsToHostStereo.
            int num_left, num_right;
            if (num_kpts_element_size_ == sizeof(int32_t))
            {
                auto *num_kpts_ptr = reinterpret_cast<const int32_t *>(
                    static_cast<const char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_);
                num_left = num_kpts_ptr[0];
                num_right = num_kpts_ptr[1];
            }
            else
            {
                auto *num_kpts_ptr = reinterpret_cast<const int64_t *>(
                    static_cast<const char *>(data_.h_pinned_buffer_) + data_.num_kpts_offset_);
                num_left = static_cast<int>(num_kpts_ptr[0]);
                num_right = static_cast<int>(num_kpts_ptr[1]);
            }

            return {std::min(num_left, params_.max_keypoints),
                    std::min(num_right, params_.max_keypoints)};
        }

        cv::cuda::GpuMat FeatureExtractor::wrapDescriptorsAsGpuMatStereo(int batch_idx, int num_keypoints) const
        {
            if (batch_idx < 0 || batch_idx >= 2 || params_.batch_size != 2)
            {
                return cv::cuda::GpuMat();
            }

            void *desc_buf = getDescriptorsGpuBuffer();
            if (!desc_buf || num_keypoints <= 0)
            {
                return cv::cuda::GpuMat();
            }

            // Descriptors are stored as (2, max_keypoints, descriptor_dim)
            // Compute offset for batch_idx
            size_t batch_offset = batch_idx * params_.max_keypoints * data_.descriptor_dim_ * sizeof(float);
            void *desc_ptr = static_cast<char *>(desc_buf) + batch_offset;

            return cv::cuda::GpuMat(
                num_keypoints,
                data_.descriptor_dim_,
                CV_32F,
                desc_ptr,
                data_.descriptor_dim_ * sizeof(float));
        }

    } // namespace perception
} // namespace uosm