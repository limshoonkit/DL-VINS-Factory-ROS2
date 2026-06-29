#include "loop_fusion/dino_vpr.hpp"
#include "loop_fusion/dino_vpr_preprocess.cuh"

#include <opencv2/core/cuda.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace uosm::loop_fusion
{
    namespace
    {
        class DinoTrtLogger : public nvinfer1::ILogger
        {
        public:
            void log(Severity severity, const char *msg) noexcept override
            {
                if (severity <= Severity::kWARNING)
                    std::cerr << "[dino_vpr][trt] " << msg << std::endl;
            }
        };
        DinoTrtLogger g_dino_logger;
    } // namespace

    DinoVpr::DinoVpr(const Params &p) : params_(p)
    {
        num_tokens_ = (params_.input_h / params_.patch) *
                      (params_.input_w / params_.patch);

        if (!vlad_.load(params_.vocab_path))
        {
            std::cerr << "[dino_vpr] failed to load vocabulary: "
                      << params_.vocab_path << std::endl;
            return;
        }
        if (vlad_.descriptorDim() != params_.embed_dim)
        {
            std::cerr << "[dino_vpr] vocab dim " << vlad_.descriptorDim()
                      << " != embed_dim " << params_.embed_dim << std::endl;
            return;
        }
        if (!initEngine())
            return;
        if (!discoverNames())
            return;
        if (!allocate())
            return;

        std::cout << "[dino_vpr] ready: " << params_.input_h << "x"
                  << params_.input_w << " -> " << num_tokens_ << " tokens x "
                  << params_.embed_dim << "  (VLAD dim " << vlad_.vladDim()
                  << ")" << std::endl;
    }

    DinoVpr::~DinoVpr()
    {
        if (stream_)
            cudaStreamSynchronize(stream_);
        context_.reset();
        engine_.reset();
        runtime_.reset();
        if (d_input_)
            cudaFree(d_input_);
        if (d_tokens_)
            cudaFree(d_tokens_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    bool DinoVpr::initEngine()
    {
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(g_dino_logger));
        if (!runtime_)
            return false;

        std::ifstream f(params_.engine_path, std::ios::binary);
        if (!f.good())
        {
            std::cerr << "[dino_vpr] engine not found: " << params_.engine_path
                      << std::endl;
            return false;
        }
        f.seekg(0, std::ios::end);
        const size_t sz = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);
        std::vector<char> blob(sz);
        f.read(blob.data(), static_cast<std::streamsize>(sz));

        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(blob.data(), sz));
        if (!engine_)
        {
            std::cerr << "[dino_vpr] deserializeCudaEngine failed" << std::endl;
            return false;
        }
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        if (!context_)
            return false;
        if (cudaStreamCreate(&stream_) != cudaSuccess)
            return false;
        return true;
    }

    bool DinoVpr::discoverNames()
    {
        const int32_t n = engine_->getNbIOTensors();
        for (int32_t i = 0; i < n; ++i)
        {
            const char *name = engine_->getIOTensorName(i);
            const auto mode = engine_->getTensorIOMode(name);
            const auto shape = engine_->getTensorShape(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                input_name_ = name;
                if (shape.nbDims >= 4)
                {
                    params_.input_h = shape.d[2];
                    params_.input_w = shape.d[3];
                }
            }
            else
            {
                output_name_ = name;
                // tokens: (1, T, embed) — recover T and embed defensively.
                if (shape.nbDims == 3)
                {
                    num_tokens_ = shape.d[1];
                    params_.embed_dim = shape.d[2];
                }
            }
        }
        // Recompute tokens from the (possibly engine-overridden) input size.
        const int t_from_size = (params_.input_h / params_.patch) *
                                (params_.input_w / params_.patch);
        if (num_tokens_ <= 0)
            num_tokens_ = t_from_size;
        if (input_name_.empty() || output_name_.empty())
        {
            std::cerr << "[dino_vpr] could not find engine I/O tensors"
                      << std::endl;
            return false;
        }
        if (params_.embed_dim != vlad_.descriptorDim())
        {
            std::cerr << "[dino_vpr] engine embed " << params_.embed_dim
                      << " != vocab dim " << vlad_.descriptorDim() << std::endl;
            return false;
        }
        return true;
    }

    bool DinoVpr::allocate()
    {
        const size_t in_bytes = static_cast<size_t>(3) * params_.input_h *
                                params_.input_w * sizeof(float);
        const size_t tok_bytes = static_cast<size_t>(num_tokens_) *
                                 params_.embed_dim * sizeof(float);
        if (cudaMalloc(reinterpret_cast<void **>(&d_input_), in_bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&d_tokens_), tok_bytes) != cudaSuccess)
        {
            std::cerr << "[dino_vpr] cudaMalloc failed" << std::endl;
            return false;
        }
        h_tokens_.assign(static_cast<size_t>(num_tokens_) * params_.embed_dim, 0.0f);

        context_->setInputShape(
            input_name_.c_str(),
            nvinfer1::Dims4{1, 3, params_.input_h, params_.input_w});
        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(output_name_.c_str(), d_tokens_);
        return true;
    }

    std::vector<float> DinoVpr::compute(const cv::Mat &image)
    {
        if (!ok() || image.empty())
            return {};

        cv::cuda::GpuMat gpu;
        gpu.upload(image);

        const float scale = std::min(
            static_cast<float>(params_.input_w) / gpu.cols,
            static_cast<float>(params_.input_h) / gpu.rows);
        const int new_w = static_cast<int>(gpu.cols * scale);
        const int new_h = static_cast<int>(gpu.rows * scale);
        const int pad_x = (params_.input_w - new_w) / 2;
        const int pad_y = (params_.input_h - new_h) / 2;

        launchDinoPreprocess(gpu, d_input_, params_.input_h, params_.input_w,
                             new_w, new_h, pad_x, pad_y, stream_);

        if (!context_->enqueueV3(stream_))
        {
            std::cerr << "[dino_vpr] enqueueV3 failed" << std::endl;
            return {};
        }

        const size_t tok_bytes = h_tokens_.size() * sizeof(float);
        if (cudaMemcpyAsync(h_tokens_.data(), d_tokens_, tok_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
            return {};
        cudaStreamSynchronize(stream_);

        const cv::Mat tokens(num_tokens_, params_.embed_dim, CV_32F,
                             h_tokens_.data());
        return vlad_.encode(tokens);
    }
} // namespace uosm::loop_fusion
