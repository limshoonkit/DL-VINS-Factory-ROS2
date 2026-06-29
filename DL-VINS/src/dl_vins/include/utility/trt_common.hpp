#ifndef UOSM_UTILITY_TRT_COMMON_HPP_
#define UOSM_UTILITY_TRT_COMMON_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace uosm
{
    namespace utility
    {
        inline void cudaCheck(cudaError_t error, const char *file, int line)
        {
            if (error != cudaSuccess)
            {
                std::cerr << "CUDA error at " << file << ":" << line
                          << " - " << cudaGetErrorString(error) << std::endl;
                std::exit(1);
            }
        }

        class TrtLogger : public nvinfer1::ILogger
        {
        public:
            void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
            {
                if (severity <= nvinfer1::ILogger::Severity::kINFO)
                    std::cout << msg << std::endl;
            }
        };

        inline TrtLogger gTrtLogger;

        // Output allocator for TRT bindings whose shape is only known after enqueue.
        // See https://github.com/NVIDIA/TensorRT/issues/4120 for the original pattern.
        class DynamicOutputAllocator : public nvinfer1::IOutputAllocator
        {
        public:
            DynamicOutputAllocator() = default;

            ~DynamicOutputAllocator() override
            {
                if (mBuffer)
                    cudaFree(mBuffer);
            }

            void notifyShape(const char * /*tensorName*/,
                             const nvinfer1::Dims &dims) noexcept override
            {
                mShape = dims;
            }

            void *reallocateOutputAsync(const char *tensorName,
                                        void * /*currentMemory*/,
                                        uint64_t size,
                                        uint64_t /*alignment*/,
                                        cudaStream_t /*stream*/) noexcept override
            {
                if (size > mSize)
                {
                    if (mBuffer)
                        cudaFree(mBuffer);
                    if (cudaMalloc(&mBuffer, size) != cudaSuccess)
                    {
                        std::cerr << "ERROR: Failed to allocate GPU memory for output tensor "
                                  << tensorName << std::endl;
                        mSize = 0;
                        mBuffer = nullptr;
                        return nullptr;
                    }
                    mSize = size;
                }
                return mBuffer;
            }

            void *getBuffer() const { return mBuffer; }
            nvinfer1::Dims getShape() const { return mShape; }

        private:
            void *mBuffer{nullptr};
            uint64_t mSize{0};
            nvinfer1::Dims mShape{};
        };

        // Description of one input tensor binding for warmup.
        struct WarmupBinding
        {
            const char *name;
            nvinfer1::Dims shape;
            void *device_ptr;
            size_t bytes;
        };

        // Run an engine for n_iters with zeroed inputs to pay first-inference costs
        // (lazy CUDA kernel loading, cuDNN/cuBLAS tactic selection, allocator first-fit)
        // up front instead of on the first real frame.
        inline bool warmupEngine(nvinfer1::IExecutionContext *context,
                                 cudaStream_t stream,
                                 const std::vector<WarmupBinding> &inputs,
                                 int n_iters = 3)
        {
            if (!context || !stream || n_iters <= 0)
                return false;

            for (const auto &b : inputs)
            {
                if (b.device_ptr && b.bytes > 0)
                {
                    if (cudaMemsetAsync(b.device_ptr, 0, b.bytes, stream) != cudaSuccess)
                        return false;
                }
                if (!context->setInputShape(b.name, b.shape))
                    return false;
                if (!context->setTensorAddress(b.name, b.device_ptr))
                    return false;
            }

            for (int i = 0; i < n_iters; ++i)
            {
                if (!context->enqueueV3(stream))
                    return false;
            }
            return cudaStreamSynchronize(stream) == cudaSuccess;
        }

        // Load a serialized TensorRT engine file.
        inline std::unique_ptr<nvinfer1::ICudaEngine> loadCudaEngine(
            const std::string &engine_path,
            std::unique_ptr<nvinfer1::IRuntime> &runtime_out,
            nvinfer1::ILogger &logger = gTrtLogger)
        {
            std::ifstream f(engine_path, std::ios::binary);
            if (!f.good())
                return nullptr;
            f.seekg(0, std::ios::end);
            const std::streamsize sz = f.tellg();
            f.seekg(0, std::ios::beg);
            if (sz <= 0)
                return nullptr;
            std::vector<char> blob(static_cast<size_t>(sz));
            if (!f.read(blob.data(), sz))
                return nullptr;

            runtime_out.reset(nvinfer1::createInferRuntime(logger));
            if (!runtime_out)
                return nullptr;
            return std::unique_ptr<nvinfer1::ICudaEngine>(
                runtime_out->deserializeCudaEngine(blob.data(), static_cast<size_t>(sz)));
        }

        // Reset TRT objects. NOTE: need in reverse-creation order (context -> engine -> runtime).
        template <typename Ctx, typename Eng, typename Rt>
        inline void resetTrtEngine(Ctx &context, Eng &engine, Rt &runtime)
        {
            context.reset();
            engine.reset();
            runtime.reset();
        }
    } // namespace utility
} // namespace uosm

#define CUDA_CHECK(call) ::uosm::utility::cudaCheck((call), __FILE__, __LINE__)

#endif // UOSM_UTILITY_TRT_COMMON_HPP_
