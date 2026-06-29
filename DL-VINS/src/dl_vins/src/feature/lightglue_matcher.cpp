#include "../include/feature/lightglue_matcher.hpp"
#include "../include/feature/type_conv_helper.cuh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

#include <NvInferPlugin.h>
#include <rclcpp/logging.hpp>

namespace uosm
{
    namespace perception
    {
        namespace
        {
            // All TRT consumers share a single logger via uosm::utility::gTrtLogger.
            inline nvinfer1::ILogger &lgLogger() { return uosm::utility::gTrtLogger; }

            constexpr const char *kLog = "lightglue_matcher";

            bool endsWith(const std::string &s, const std::string &suffix)
            {
                if (suffix.size() > s.size())
                    return false;
                return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
            }
        } // namespace

        const std::string LightGlueMatcher::kName = "lightglue";

        LightGlueMatcher::LightGlueMatcher(Params params) : params_(std::move(params))
        {
            if (!init())
            {
                RCLCPP_WARN(rclcpp::get_logger(kLog),
                            "LightGlueMatcher initialization failed; match() will return empty results. "
                            "engine_path='%s'",
                            params_.engine_path.c_str());
                shutdown();
            }
        }

        LightGlueMatcher::~LightGlueMatcher()
        {
            shutdown();
        }

        void LightGlueMatcher::shutdownSlot(MatcherSlot &slot)
        {
            if (slot.stream)
                cudaStreamSynchronize(slot.stream);
            slot.allocators.clear();
            if (slot.context)
                slot.context.reset();
            if (slot.d_kpts0)
            {
                cudaFree(slot.d_kpts0);
                slot.d_kpts0 = nullptr;
            }
            if (slot.d_kpts1)
            {
                cudaFree(slot.d_kpts1);
                slot.d_kpts1 = nullptr;
            }
            slot.d_kpts0_capacity_bytes = 0;
            slot.d_kpts1_capacity_bytes = 0;
            if (slot.stream)
            {
                cudaStreamDestroy(slot.stream);
                slot.stream = nullptr;
            }
        }

        void LightGlueMatcher::shutdown()
        {
            shutdownSlot(slot_a_);

            for (void **p : {reinterpret_cast<void **>(&d_batch_kpts0_),
                             reinterpret_cast<void **>(&d_batch_kpts1_),
                             &d_batch_desc0_, &d_batch_desc1_})
            {
                if (*p)
                {
                    cudaFree(*p);
                    *p = nullptr;
                }
            }

            if (engine_)
                engine_.reset();
            if (runtime_)
                runtime_.reset();
        }

        bool LightGlueMatcher::init()
        {
            if (params_.engine_path.empty())
            {
                RCLCPP_WARN(rclcpp::get_logger(kLog),
                            "LightGlueMatcher engine_path is empty; matcher disabled.");
                return false;
            }

            std::ifstream file(params_.engine_path, std::ios::binary);
            if (!file.good())
            {
                RCLCPP_WARN(rclcpp::get_logger(kLog),
                            "LightGlue engine file not found at '%s'; matcher disabled.",
                            params_.engine_path.c_str());
                return false;
            }

            if (!initLibNvInferPlugins(&lgLogger(), ""))
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "Failed to initialize TensorRT plugins.");
                return false;
            }

            engine_ = uosm::utility::loadCudaEngine(params_.engine_path, runtime_, lgLogger());
            if (!engine_)
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "Failed to deserialize LightGlue engine.");
                return false;
            }

            if (!discoverTensorNames())
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog),
                             "Could not map expected LightGlue I/O tensors on this engine.");
                return false;
            }

            detectBatched();

            if (!initSlot(slot_a_))
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "Failed to initialize execution slot.");
                return false;
            }

            RCLCPP_INFO(rclcpp::get_logger(kLog),
                        "LightGlueMatcher initialized: engine='%s', desc_dim=%d, max_kp=%d, "
                        "kpt_dim=%d, threshold=%.2f, mode=%s",
                        params_.engine_path.c_str(), params_.descriptor_dim,
                        params_.max_keypoints, kpt_dim_, params_.score_threshold,
                        batched_ ? ("batched (cap=" + std::to_string(batch_capacity_) + ")").c_str()
                                 : "single-pair");

            warmup(3);
            return true;
        }

        void LightGlueMatcher::detectBatched()
        {
            batched_ = false;
            batch_capacity_ = 1;

            auto mshape = engine_->getTensorShape(matches_name_.c_str());
            if (mshape.nbDims >= 1 && mshape.d[mshape.nbDims - 1] == 3)
                batched_ = true;
            if (!batched_)
                return;

            const int nb_profiles = engine_->getNbOptimizationProfiles();
            if (nb_profiles > 0)
            {
                auto kmax = engine_->getProfileShape(kpts0_name_.c_str(), 0,
                                                     nvinfer1::OptProfileSelector::kMAX);
                if (kmax.nbDims >= 1 && kmax.d[0] > 0)
                    batch_capacity_ = static_cast<int>(kmax.d[0]);
            }
        }

        bool LightGlueMatcher::initSlot(MatcherSlot &slot)
        {
            slot.context = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
            if (!slot.context)
                return false;

            if (cudaStreamCreate(&slot.stream) != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "Failed to create CUDA stream for slot.");
                slot.stream = nullptr;
                return false;
            }

            for (const auto &name : {matches_name_, mscores_name_})
            {
                if (name.empty())
                    continue;
                auto alloc = std::make_unique<uosm::utility::DynamicOutputAllocator>();
                if (!slot.context->setOutputAllocator(name.c_str(), alloc.get()))
                {
                    RCLCPP_ERROR(rclcpp::get_logger(kLog),
                                 "Failed to set output allocator for tensor '%s'", name.c_str());
                    return false;
                }
                slot.allocators.emplace(name, std::move(alloc));
            }
            return true;
        }

        bool LightGlueMatcher::warmup(int n_iters)
        {
            if (!is_initialized())
                return false;

            if (batched_)
            {
                if (!ensureBatchBuffers())
                {
                    RCLCPP_WARN(rclcpp::get_logger(kLog), "Warmup: failed to allocate batch buffers");
                    return false;
                }
                const int B = batch_capacity_;
                const int max_kp = params_.max_keypoints;
                const int D = params_.descriptor_dim;
                const size_t kpts_bytes = static_cast<size_t>(B) * max_kp * kpt_dim_ * sizeof(float);
                const size_t desc_bytes = static_cast<size_t>(B) * max_kp * D * sizeof(float);
                std::vector<uosm::utility::WarmupBinding> bindings = {
                    {kpts0_name_.c_str(), nvinfer1::Dims3{B, max_kp, kpt_dim_}, d_batch_kpts0_, kpts_bytes},
                    {kpts1_name_.c_str(), nvinfer1::Dims3{B, max_kp, kpt_dim_}, d_batch_kpts1_, kpts_bytes},
                    {desc0_name_.c_str(), nvinfer1::Dims3{B, max_kp, D}, d_batch_desc0_, desc_bytes},
                    {desc1_name_.c_str(), nvinfer1::Dims3{B, max_kp, D}, d_batch_desc1_, desc_bytes},
                };
                if (!uosm::utility::warmupEngine(slot_a_.context.get(), slot_a_.stream,
                                                 bindings, n_iters))
                {
                    RCLCPP_WARN(rclcpp::get_logger(kLog), "Warmup failed");
                    return false;
                }
                RCLCPP_INFO(rclcpp::get_logger(kLog),
                            "Warmup complete (%d iters, batched B=%d, max_kp=%d, desc_dim=%d)",
                            n_iters, B, max_kp, D);
                return true;
            }

            if (!ensureSlotCapacity(slot_a_))
            {
                RCLCPP_WARN(rclcpp::get_logger(kLog), "Warmup: failed to allocate kpts buffers");
                return false;
            }

            const int max_kp = params_.max_keypoints;
            const size_t kpts_bytes = static_cast<size_t>(max_kp) * kpt_dim_ * sizeof(float);
            const size_t desc_bytes = static_cast<size_t>(max_kp) * params_.descriptor_dim * sizeof(float);

            // Temp descriptor scratch -- worst-case max_kp so TRT picks kernels for the
            // longest input path.
            void *d_desc0_tmp = nullptr;
            void *d_desc1_tmp = nullptr;
            if (cudaMalloc(&d_desc0_tmp, desc_bytes) != cudaSuccess ||
                cudaMalloc(&d_desc1_tmp, desc_bytes) != cudaSuccess)
            {
                if (d_desc0_tmp)
                    cudaFree(d_desc0_tmp);
                if (d_desc1_tmp)
                    cudaFree(d_desc1_tmp);
                RCLCPP_WARN(rclcpp::get_logger(kLog), "Warmup: failed to allocate temp descriptor buffers");
                return false;
            }

            std::vector<uosm::utility::WarmupBinding> bindings = {
                {kpts0_name_.c_str(), nvinfer1::Dims3{1, max_kp, kpt_dim_}, slot_a_.d_kpts0, kpts_bytes},
                {kpts1_name_.c_str(), nvinfer1::Dims3{1, max_kp, kpt_dim_}, slot_a_.d_kpts1, kpts_bytes},
                {desc0_name_.c_str(), nvinfer1::Dims3{1, max_kp, params_.descriptor_dim}, d_desc0_tmp, desc_bytes},
                {desc1_name_.c_str(), nvinfer1::Dims3{1, max_kp, params_.descriptor_dim}, d_desc1_tmp, desc_bytes},
            };
            const bool ok = uosm::utility::warmupEngine(slot_a_.context.get(), slot_a_.stream,
                                                        bindings, n_iters);

            cudaFree(d_desc0_tmp);
            cudaFree(d_desc1_tmp);

            if (!ok)
            {
                RCLCPP_WARN(rclcpp::get_logger(kLog), "Warmup failed");
                return false;
            }
            RCLCPP_INFO(rclcpp::get_logger(kLog),
                        "Warmup complete (%d iters, max_kp=%d, desc_dim=%d)",
                        n_iters, max_kp, params_.descriptor_dim);
            return true;
        }

        bool LightGlueMatcher::discoverTensorNames()
        {
            const int nbIOTensors = engine_->getNbIOTensors();
            for (int i = 0; i < nbIOTensors; ++i)
            {
                std::string name = engine_->getIOTensorName(i);
                const auto mode = engine_->getTensorIOMode(name.c_str());
                if (mode == nvinfer1::TensorIOMode::kINPUT)
                {
                    if (endsWith(name, "kpts0") || name == "kpts0")
                        kpts0_name_ = name;
                    else if (endsWith(name, "kpts1") || name == "kpts1")
                        kpts1_name_ = name;
                    else if (endsWith(name, "desc0") || name == "desc0")
                        desc0_name_ = name;
                    else if (endsWith(name, "desc1") || name == "desc1")
                        desc1_name_ = name;
                }
                else
                {
                    if (name.find("match") != std::string::npos && matches_name_.empty())
                        matches_name_ = name;
                    else if (name.find("score") != std::string::npos && mscores_name_.empty())
                        mscores_name_ = name;
                }
            }

            if (kpts0_name_.empty() || kpts1_name_.empty() ||
                desc0_name_.empty() || desc1_name_.empty() ||
                matches_name_.empty())
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog),
                             "LightGlue tensor mapping incomplete: kpts0='%s' kpts1='%s' "
                             "desc0='%s' desc1='%s' matches='%s' mscores='%s'",
                             kpts0_name_.c_str(), kpts1_name_.c_str(),
                             desc0_name_.c_str(), desc1_name_.c_str(),
                             matches_name_.c_str(), mscores_name_.c_str());
                return false;
            }

            auto kshape = engine_->getTensorShape(kpts0_name_.c_str());
            if (kshape.nbDims >= 3 && kshape.d[2] > 0)
                kpt_dim_ = static_cast<int>(kshape.d[2]);
            else if (engine_->getNbOptimizationProfiles() > 0)
            {
                auto kopt = engine_->getProfileShape(kpts0_name_.c_str(), 0,
                                                     nvinfer1::OptProfileSelector::kOPT);
                if (kopt.nbDims >= 3 && kopt.d[2] > 0)
                    kpt_dim_ = static_cast<int>(kopt.d[2]);
            }
            return true;
        }

        bool LightGlueMatcher::ensureSlotCapacity(MatcherSlot &slot)
        {
            const int max_kp = params_.max_keypoints;
            const size_t kpts_bytes = static_cast<size_t>(max_kp) * kpt_dim_ * sizeof(float);

            auto grow = [](float *&p, size_t &cap, size_t need) -> bool
            {
                if (need <= cap && p != nullptr)
                    return true;
                if (p)
                {
                    cudaFree(p);
                    p = nullptr;
                }
                if (cudaMalloc(reinterpret_cast<void **>(&p), need) != cudaSuccess)
                {
                    cap = 0;
                    p = nullptr;
                    return false;
                }
                cap = need;
                return true;
            };

            if (!grow(slot.d_kpts0, slot.d_kpts0_capacity_bytes, kpts_bytes))
                return false;
            if (!grow(slot.d_kpts1, slot.d_kpts1_capacity_bytes, kpts_bytes))
                return false;
            return true;
        }

        // Allocate the (batch_capacity_, max_kp, *) staging buffers used by
        // matchBatched(). Sized once for the engine's max batch; reused every call.
        bool LightGlueMatcher::ensureBatchBuffers()
        {
            if (d_batch_kpts0_ && d_batch_kpts1_ && d_batch_desc0_ && d_batch_desc1_)
                return true;

            const size_t B = static_cast<size_t>(std::max(batch_capacity_, 1));
            const size_t max_kp = static_cast<size_t>(params_.max_keypoints);
            const size_t D = static_cast<size_t>(params_.descriptor_dim);
            const size_t kpts_bytes = B * max_kp * kpt_dim_ * sizeof(float);
            const size_t desc_bytes = B * max_kp * D * sizeof(float);

            if (cudaMalloc(reinterpret_cast<void **>(&d_batch_kpts0_), kpts_bytes) != cudaSuccess ||
                cudaMalloc(reinterpret_cast<void **>(&d_batch_kpts1_), kpts_bytes) != cudaSuccess ||
                cudaMalloc(&d_batch_desc0_, desc_bytes) != cudaSuccess ||
                cudaMalloc(&d_batch_desc1_, desc_bytes) != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog),
                             "ensureBatchBuffers(): failed to allocate batch staging buffers");
                return false;
            }
            return true;
        }

        // Launch normalize + setInput + enqueueV3 on the slot's stream. Does NOT sync.
        // n0_use_out / n1_use_out are filled with the clamped counts so collect knows
        // them for index-validity filtering.
        bool LightGlueMatcher::enqueuePairAsync(MatcherSlot &slot, const MatchInputs &in,
                                           int &n0_use_out, int &n1_use_out)
        {
            if (!is_initialized() || !in.d_desc0 || !in.d_desc1 ||
                !in.d_kpts0_px || !in.d_kpts1_px || in.n0 <= 0 || in.n1 <= 0)
                return false;
            if (!ensureSlotCapacity(slot))
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog),
                             "enqueuePairAsync(): failed to allocate slot input buffers");
                return false;
            }

            const int max_kp = params_.max_keypoints;
            const int n0_use = std::min(in.n0, max_kp);
            const int n1_use = std::min(in.n1, max_kp);
            n0_use_out = n0_use;
            n1_use_out = n1_use;
            if (in.n0 > max_kp || in.n1 > max_kp)
            {
                RCLCPP_WARN(rclcpp::get_logger(kLog),
                            "enqueuePairAsync(): kpt count (%d,%d) exceeds engine max_kp=%d; extra kpts ignored.",
                            in.n0, in.n1, max_kp);
            }

            const float cx = in.image_w * 0.5f;
            const float cy = in.image_h * 0.5f;
            const float inv_scale = 1.0f / (0.5f * static_cast<float>(std::max(in.image_w, in.image_h)));

            launchNormalizeKeypointsLightGlue(in.d_kpts0_px, slot.d_kpts0, n0_use, max_kp, kpt_dim_,
                                              cx, cy, inv_scale, slot.stream);
            launchNormalizeKeypointsLightGlue(in.d_kpts1_px, slot.d_kpts1, n1_use, max_kp, kpt_dim_,
                                              cx, cy, inv_scale, slot.stream);

            auto setInput = [&](const std::string &name, const nvinfer1::Dims &dims, const void *dev_ptr) -> bool
            {
                if (!slot.context->setInputShape(name.c_str(), dims))
                    return false;
                return slot.context->setTensorAddress(name.c_str(), const_cast<void *>(dev_ptr));
            };

            const int D = params_.descriptor_dim;
            nvinfer1::Dims3 kpts_dims{1, max_kp, kpt_dim_};
            nvinfer1::Dims3 desc_dims{1, max_kp, D};
            if (!setInput(kpts0_name_, kpts_dims, slot.d_kpts0) ||
                !setInput(kpts1_name_, kpts_dims, slot.d_kpts1) ||
                !setInput(desc0_name_, desc_dims, in.d_desc0) ||
                !setInput(desc1_name_, desc_dims, in.d_desc1))
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "enqueuePairAsync(): setInputShape/Address failed");
                return false;
            }

            if (!slot.context->enqueueV3(slot.stream))
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "enqueuePairAsync(): enqueueV3 failed");
                return false;
            }
            return true;
        }

        // Sync the slot's stream and read matches/scores from its allocators.
        std::vector<DescriptorMatch>
        LightGlueMatcher::collectPairAfterSync(MatcherSlot &slot, int n0_use, int n1_use)
        {
            std::vector<DescriptorMatch> out;
            if (cudaStreamSynchronize(slot.stream) != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "collectPairAfterSync(): stream sync failed");
                return out;
            }

            auto it_m = slot.allocators.find(matches_name_);
            if (it_m == slot.allocators.end())
                return out;
            auto &matches_alloc = it_m->second;
            auto matches_shape = matches_alloc->getShape();
            if (matches_shape.nbDims < 1)
                return out;

            int num_matches = matches_shape.d[0];
            int matches_stride = 2;
            if (matches_shape.nbDims == 3 && matches_shape.d[0] == 1)
                num_matches = matches_shape.d[1];

            if (num_matches <= 0 || matches_alloc->getBuffer() == nullptr)
                return out;

            std::vector<int64_t> h_matches(num_matches * matches_stride);
            if (cudaMemcpy(h_matches.data(), matches_alloc->getBuffer(),
                           h_matches.size() * sizeof(int64_t),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return out;

            std::vector<float> h_scores;
            if (!mscores_name_.empty() && slot.allocators.count(mscores_name_))
            {
                auto &scores_alloc = slot.allocators.at(mscores_name_);
                auto scores_shape = scores_alloc->getShape();
                int num_scores = num_matches;
                if (scores_shape.nbDims >= 1)
                    num_scores = scores_shape.d[scores_shape.nbDims - 1];
                h_scores.resize(num_scores);
                if (scores_alloc->getBuffer() != nullptr && num_scores > 0)
                {
                    cudaMemcpy(h_scores.data(), scores_alloc->getBuffer(),
                               h_scores.size() * sizeof(float), cudaMemcpyDeviceToHost);
                }
            }

            out.reserve(num_matches);
            for (int i = 0; i < num_matches; ++i)
            {
                const int i0 = static_cast<int>(h_matches[i * matches_stride + 0]);
                const int i1 = static_cast<int>(h_matches[i * matches_stride + 1]);
                if (i0 < 0 || i1 < 0 || i0 >= n0_use || i1 >= n1_use)
                    continue;
                const float s = (i < static_cast<int>(h_scores.size())) ? h_scores[i] : 1.0f;
                if (s < params_.score_threshold)
                    continue;
                out.push_back({i0, i1, s});
            }
            return out;
        }

        std::vector<DescriptorMatch>
        LightGlueMatcher::runPairOnSlot(MatcherSlot &slot, const MatchInputs &in)
        {
            int n0_use = 0, n1_use = 0;
            if (!enqueuePairAsync(slot, in, n0_use, n1_use))
                return {};
            return collectPairAfterSync(slot, n0_use, n1_use);
        }

        std::vector<DescriptorMatch>
        LightGlueMatcher::match(const float *d_kpts0_px, int n0,
                                   const void *d_desc0,
                                   const float *d_kpts1_px, int n1,
                                   const void *d_desc1,
                                   int image_h, int image_w)
        {
            MatchInputs in{d_kpts0_px, n0, d_desc0,
                              d_kpts1_px, n1, d_desc1,
                              image_h, image_w};
            if (batched_)
            {
                auto r = matchBatched({in});
                return r.empty() ? std::vector<DescriptorMatch>{} : std::move(r[0]);
            }
            return runPairOnSlot(slot_a_, in);
        }

        // Batched engine: stage every pair contiguously as (B,max_kp,*), run one
        // enqueue, sync, then split the (M,3) matches output [batch_idx, idx0, idx1]
        // into per-pair vectors.
        //
        // Non-batched engine (or B exceeds the engine's batch capacity): fall back to
        // a per-pair match() loop on slot_a_. This path is structurally unreachable
        // with the production stereo `*_b3` engines but is kept for robustness if a
        // non-batched engine is configured.
        std::vector<std::vector<DescriptorMatch>>
        LightGlueMatcher::matchBatched(const std::vector<MatchInputs> &pairs)
        {
            const int B = static_cast<int>(pairs.size());
            std::vector<std::vector<DescriptorMatch>> out(std::max(B, 0));
            if (!is_initialized() || B <= 0)
                return out;

            if (!batched_ || B > batch_capacity_)
            {
                for (int i = 0; i < B; ++i)
                    out[i] = runPairOnSlot(slot_a_, pairs[i]);
                return out;
            }

            if (!ensureBatchBuffers())
                return out;

            const int max_kp = params_.max_keypoints;
            const int D = params_.descriptor_dim;
            const size_t kpts_stride = static_cast<size_t>(max_kp) * kpt_dim_;
            const size_t desc_row_bytes = static_cast<size_t>(max_kp) * D * sizeof(float);
            MatcherSlot &slot = slot_a_;

            std::vector<int> n0_use(B), n1_use(B);
            for (int b = 0; b < B; ++b)
            {
                const auto &in = pairs[b];
                if (!in.d_desc0 || !in.d_desc1 || !in.d_kpts0_px || !in.d_kpts1_px ||
                    in.n0 <= 0 || in.n1 <= 0)
                {
                    RCLCPP_ERROR(rclcpp::get_logger(kLog),
                                 "matchBatched(): pair %d has invalid inputs", b);
                    return out;
                }
                n0_use[b] = std::min(in.n0, max_kp);
                n1_use[b] = std::min(in.n1, max_kp);

                const float cx = in.image_w * 0.5f;
                const float cy = in.image_h * 0.5f;
                const float inv_scale =
                    1.0f / (0.5f * static_cast<float>(std::max(in.image_w, in.image_h)));
                launchNormalizeKeypointsLightGlue(in.d_kpts0_px, d_batch_kpts0_ + b * kpts_stride,
                                                  n0_use[b], max_kp, kpt_dim_, cx, cy, inv_scale, slot.stream);
                launchNormalizeKeypointsLightGlue(in.d_kpts1_px, d_batch_kpts1_ + b * kpts_stride,
                                                  n1_use[b], max_kp, kpt_dim_, cx, cy, inv_scale, slot.stream);
                cudaMemcpyAsync(static_cast<char *>(d_batch_desc0_) + b * desc_row_bytes,
                                in.d_desc0, desc_row_bytes, cudaMemcpyDeviceToDevice, slot.stream);
                cudaMemcpyAsync(static_cast<char *>(d_batch_desc1_) + b * desc_row_bytes,
                                in.d_desc1, desc_row_bytes, cudaMemcpyDeviceToDevice, slot.stream);
            }

            const nvinfer1::Dims3 kpts_dims{B, max_kp, kpt_dim_};
            const nvinfer1::Dims3 desc_dims{B, max_kp, D};
            auto setInput = [&](const std::string &name, const nvinfer1::Dims &dims,
                                void *dev_ptr) -> bool
            {
                return slot.context->setInputShape(name.c_str(), dims) &&
                       slot.context->setTensorAddress(name.c_str(), dev_ptr);
            };
            if (!setInput(kpts0_name_, kpts_dims, d_batch_kpts0_) ||
                !setInput(kpts1_name_, kpts_dims, d_batch_kpts1_) ||
                !setInput(desc0_name_, desc_dims, d_batch_desc0_) ||
                !setInput(desc1_name_, desc_dims, d_batch_desc1_))
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog),
                             "matchBatched(): setInputShape/Address failed");
                return out;
            }
            if (!slot.context->enqueueV3(slot.stream))
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "matchBatched(): enqueueV3 failed");
                return out;
            }
            if (cudaStreamSynchronize(slot.stream) != cudaSuccess)
            {
                RCLCPP_ERROR(rclcpp::get_logger(kLog), "matchBatched(): stream sync failed");
                return out;
            }

            auto it_m = slot.allocators.find(matches_name_);
            if (it_m == slot.allocators.end())
                return out;
            auto matches_shape = it_m->second->getShape();
            if (matches_shape.nbDims < 1)
                return out;
            int num_matches = matches_shape.d[0];
            if (matches_shape.nbDims == 3 && matches_shape.d[0] == 1)
                num_matches = matches_shape.d[1];
            if (num_matches <= 0 || it_m->second->getBuffer() == nullptr)
                return out;

            // Batched matches are (M,3): [batch_idx, idx0, idx1].
            std::vector<int64_t> h_matches(static_cast<size_t>(num_matches) * 3);
            if (cudaMemcpy(h_matches.data(), it_m->second->getBuffer(),
                           h_matches.size() * sizeof(int64_t),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return out;

            std::vector<float> h_scores;
            if (!mscores_name_.empty() && slot.allocators.count(mscores_name_))
            {
                auto &scores_alloc = slot.allocators.at(mscores_name_);
                auto scores_shape = scores_alloc->getShape();
                int num_scores = num_matches;
                if (scores_shape.nbDims >= 1)
                    num_scores = scores_shape.d[scores_shape.nbDims - 1];
                if (num_scores > 0 && scores_alloc->getBuffer() != nullptr)
                {
                    h_scores.resize(num_scores);
                    cudaMemcpy(h_scores.data(), scores_alloc->getBuffer(),
                               h_scores.size() * sizeof(float), cudaMemcpyDeviceToHost);
                }
            }

            for (int i = 0; i < num_matches; ++i)
            {
                const int b = static_cast<int>(h_matches[i * 3 + 0]);
                const int i0 = static_cast<int>(h_matches[i * 3 + 1]);
                const int i1 = static_cast<int>(h_matches[i * 3 + 2]);
                if (b < 0 || b >= B)
                    continue;
                if (i0 < 0 || i1 < 0 || i0 >= n0_use[b] || i1 >= n1_use[b])
                    continue;
                const float s = (i < static_cast<int>(h_scores.size())) ? h_scores[i] : 1.0f;
                if (s < params_.score_threshold)
                    continue;
                out[b].push_back({i0, i1, s});
            }
            return out;
        }

    } // namespace perception
} // namespace uosm
