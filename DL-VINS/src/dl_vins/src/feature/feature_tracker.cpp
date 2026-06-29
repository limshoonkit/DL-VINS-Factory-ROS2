#include "../include/feature/feature_tracker.hpp"
#include "../include/feature/triangulation.hpp"
#include "../include/feature/type_conv_helper.cuh"

#include <cstdio>
#include <filesystem>
#include <future>

FeatureTracker::ExtractorProfile FeatureTracker::profileFor(FeatureExtractionMethod method)
{
    switch (method)
    {
    case FeatureExtractionMethod::ALIKED:
        return {"aliked", 3, true};
    case FeatureExtractionMethod::SuperPoint:
        return {"superpoint", 1, true};
    case FeatureExtractionMethod::RACO:
        return {"raco", 3, true};
    case FeatureExtractionMethod::XFeat:
        return {"xfeat", 1, true};
    case FeatureExtractionMethod::SIFT_CPU:
        return {"sift", 1, true};
    default:
        return {"", 0, false};
    }
}

FeatureTracker::FeatureTracker(const FeatureTrackerParams &p)
    : enable_visualization_(p.enable_visualization),
      allow_flowback_(p.allow_flowback),
      optflow_max_iterations_(p.optflow_max_iterations),
      optflow_pyramid_levels_(p.optflow_pyramid_levels),
      optflow_window_dim_(p.optflow_window_dim),
      optflow_pyramid_scale_(p.optflow_pyramid_scale),
      optflow_epsilon_(p.optflow_epsilon),
      row_(p.row),
      col_(p.col),
      n_id_(0),
      max_tracked_keypoints_(p.max_tracked_keypoints),
      min_feature_distance_(p.min_feature_distance),
      prev_time_(0.0),
      cur_time_(0.0),
      prev_img_(cv::Mat::zeros(p.row, p.col, CV_8UC1)),
      cur_img0_(cv::Mat::zeros(p.row, p.col, CV_8UC1)),
      cur_img1_(),
      mask_(cv::Mat::zeros(p.row, p.col, CV_8UC1)),
      tracked_image_(cv::Mat::zeros(p.row, p.col, CV_8UC1)),
      n_pts_(std::vector<cv::Point2f>()),
      prev_pts0_(std::vector<cv::Point2f>()),
      cur_pts0_(std::vector<cv::Point2f>()),
      cur_pts1_(std::vector<cv::Point2f>()),
      cur_undistorted_pts0_(std::vector<cv::Point2f>()),
      cur_undistorted_pts1_(std::vector<cv::Point2f>()),
      pts_vel0_(std::vector<cv::Point2f>()),
      pts_vel1_(std::vector<cv::Point2f>()),
      ids0_(std::vector<int>()),
      ids1_(std::vector<int>()),
      track_cnt_(std::vector<int>()),
      track_cnt1_(std::vector<int>()),
      feature_extraction_method_(p.feature_extraction_method),
      optflow_device_(p.optflow_device),
      weights_folder_(p.weights_folder),
      use_descriptor_matcher_(p.use_descriptor_matcher),
      matcher_type_(p.matcher_type),
      matcher_score_threshold_(p.matcher_score_threshold),
      profile_trt_inference_(p.profile_trt_inference),
      metrics_logger_(uosm::utility::MetricsLogger::getInstance()),
      stereo_window_dim_(p.stereo_window_dim),
      stereo_pyramid_levels_(p.stereo_pyramid_levels),
      stereo_baseline_(0.0f),
      focal_length_(0.0f),
      raco_cov_alpha_(p.raco_cov_alpha),
      raco_cov_floor_px_(p.raco_cov_floor_px)
{
    // Warn about suboptimal CPU/GPU mixing
    if (feature_extraction_method_ == FeatureExtractionMethod::GFTT_CPU &&
        optflow_device_ == OpticalFlowBackend::CUDA)
    {
        RCLCPP_WARN(rclcpp::get_logger("feature_tracker"),
                    "Using CPU GFTT with CUDA optical flow is suboptimal. "
                    "Consider using GFTT_CUDA for better performance.");
    }
    if (feature_extraction_method_ == FeatureExtractionMethod::GFTT_CUDA &&
        optflow_device_ == OpticalFlowBackend::CPU)
    {
        RCLCPP_WARN(rclcpp::get_logger("feature_tracker"),
                    "Using CUDA GFTT with CPU optical flow is suboptimal. "
                    "Consider using CPU GFTT or CUDA optical flow for better performance.");
    }

    // Load static FOV mask (for fisheye/omnidirectional cameras). Mono only.
    if (p.fisheye && !p.fisheye_mask.empty())
    {
        base_mask_ = cv::imread(p.fisheye_mask, cv::IMREAD_GRAYSCALE);
        if (base_mask_.empty())
            throw std::runtime_error("Failed to load fisheye mask: " + p.fisheye_mask);
        if (base_mask_.cols != col_ || base_mask_.rows != row_)
            cv::resize(base_mask_, base_mask_, cv::Size(col_, row_), 0, 0, cv::INTER_NEAREST);
        RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                    "Loaded fisheye mask (%dx%d): %s", col_, row_, p.fisheye_mask.c_str());
    }

    // use_descriptor_matcher requires DL descriptors; force-disable it for incompatible
    // extractors / interfaces so we fall back cleanly to KLT instead of crashing later.
    const auto extractor_profile = FeatureTracker::profileFor(feature_extraction_method_);
    if (use_descriptor_matcher_ && !extractor_profile.is_dl)
    {
        RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                     "use_descriptor_matcher=true requires a DL extractor (ALIKED/SuperPoint/RACO/XFeat); "
                     "got non-DL mode. Disabling descriptor matcher.");
        use_descriptor_matcher_ = false;
    }
    // Init feature extractor, optical flow, etc
    if (!p.calib_file_cam0.empty() && readIntrinsicParameter(p.calib_file_cam0, m_cameras_[0]) == false)
    {
        throw std::runtime_error("Cannot read camera intrinsics from " + p.calib_file_cam0);
    }
    if (!p.calib_file_cam1.empty() && readIntrinsicParameter(p.calib_file_cam1, m_cameras_[1]) == false)
    {
        throw std::runtime_error("Cannot read camera intrinsics from " + p.calib_file_cam1);
    }

    // Extract focal length from camera intrinsics
    if (m_cameras_[0])
    {
        // PinholeCamera
        auto pinhole = boost::dynamic_pointer_cast<camodocal::PinholeCamera>(m_cameras_[0]);
        if (pinhole)
        {
            focal_length_ = static_cast<float>((pinhole->getParameters().fx() + pinhole->getParameters().fy()) / 2.0);
            RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                        "Auto-computed focal_length from PinholeCamera: %.1fpx", focal_length_);
        }
        else
        {
            // CataCamera (MEI model)
            // MEI projection: u = gamma * X / (Z + xi*||P||), near center ≈ gamma / (1 + xi)
            auto cata = boost::dynamic_pointer_cast<camodocal::CataCamera>(m_cameras_[0]);
            if (cata)
            {
                double xi = cata->getParameters().xi();
                double gamma_avg = (cata->getParameters().gamma1() + cata->getParameters().gamma2()) / 2.0;
                focal_length_ = static_cast<float>(gamma_avg / (1.0 + xi));
                RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                            "Auto-computed focal_length from CataCamera (MEI): %.1fpx "
                            "(gamma=%.1f, xi=%.3f, effective=gamma/(1+xi))",
                            focal_length_, gamma_avg, xi);
            }
            else
            {
                // Try EquidistantCamera
                auto equi = boost::dynamic_pointer_cast<camodocal::EquidistantCamera>(m_cameras_[0]);
                if (equi)
                {
                    focal_length_ = static_cast<float>((equi->getParameters().mu() + equi->getParameters().mv()) / 2.0);
                    RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                                "Auto-computed focal_length from EquidistantCamera: %.1fpx", focal_length_);
                }
                else
                {
                    RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                                 "Unknown camera model, focal_length not auto-computed! This will cause issues.");
                }
            }
        }
    }

    use_stereo_mode_ = !p.calib_file_cam1.empty();

    // Initialize CUDA optical flow (only when KLT is the tracker).
    if (!use_descriptor_matcher_ && optflow_device_ == OpticalFlowBackend::CUDA)
    {
        initCUDAOptflow();
    }

    // Initialize CUDA GFTT
    if (feature_extraction_method_ == FeatureExtractionMethod::GFTT_CUDA)
    {
        cuda_gftt_detector_ = cv::cuda::createGoodFeaturesToTrackDetector(
            CV_8UC1, max_tracked_keypoints_, 0.01, min_feature_distance_);
    }

    // Initialize DL feature extractor
    if (extractor_profile.is_dl)
    {
        const std::string &model_name = extractor_profile.model_name;
        const std::string role = use_stereo_mode_ ? "stereo_extractor" : "mono_extractor";
        const int requested_batch = use_stereo_mode_ ? 2 : 1;

        if (feature_extraction_method_ == FeatureExtractionMethod::SIFT_CPU)
        {
            // OpenCV CPU SIFT; allocate GPU kp+desc buffers.
            constexpr int D = 128;
            constexpr int KPT = 4; // xy + scale + orientation (radians)
            const size_t desc_bytes = 2ULL * max_tracked_keypoints_ * D * sizeof(float);
            const size_t kps_bytes = 2ULL * max_tracked_keypoints_ * KPT * sizeof(float);
            if (cudaMalloc(reinterpret_cast<void **>(&d_sift_desc_), desc_bytes) != cudaSuccess ||
                cudaMalloc(reinterpret_cast<void **>(&d_sift_kps_), kps_bytes) != cudaSuccess)
                throw std::runtime_error("SIFT_CPU: cudaMalloc descriptor/keypoint buffers failed");
            cudaMemset(d_sift_desc_, 0, desc_bytes);
            cudaMemset(d_sift_kps_, 0, kps_bytes);
            d_sift_desc_cap_ = desc_bytes;
            d_sift_kps_cap_ = kps_bytes;
            sift_detector_ = cv::SIFT::create(max_tracked_keypoints_);
            RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                        "SIFT_CPU: GPU buffers allocated (%zu + %zu bytes)",
                        desc_bytes, kps_bytes);
        }
        else if (feature_extraction_method_ == FeatureExtractionMethod::ALIKED)
        {
            // ALIKED: dense extractor + custom CUDA DKD + (optional) dhead.
            if (!initDenseAlikedComponents(requested_batch,
                                           /*need_dhead=*/use_descriptor_matcher_))
                throw std::runtime_error("ALIKED: failed to initialise dense+dhead pipeline");
            RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                        "Loaded ALIKED dense pipeline (batch=%d, dhead=%s)",
                        requested_batch, use_descriptor_matcher_ ? "yes" : "no (LK)");
        }
        else
        {
            // SuperPoint, RaCo, XFeat: single-engine FeatureExtractor.
            std::vector<std::string> extractor_tags{model_name};
            std::string resolved_engine = resolveEnginePath(role, extractor_tags,
                                                            max_tracked_keypoints_,
                                                            {matcher_type_});
            if (resolved_engine.empty())
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                             "Could not resolve %s engine (model='%s', kp=%d) under weights_folder='%s'",
                             role.c_str(), model_name.c_str(), max_tracked_keypoints_,
                             weights_folder_.c_str());
                throw std::runtime_error("DL extractor engine not found under " + weights_folder_ + "/" + role);
            }

            uosm::perception::FeatureExtractor::FeatureExtractorParams params;
            params.input_width = col_;
            params.input_height = row_;
            params.max_keypoints = max_tracked_keypoints_;
            params.profile_inference = profile_trt_inference_;
            params.use_unified_memory = false;
            params.model_type = model_name;
            params.input_channels = extractor_profile.input_channels;
            params.batch_size = use_stereo_mode_ ? 2 : 1;
            params.engine_path = resolved_engine;

            feature_extractor_ = std::make_unique<uosm::perception::FeatureExtractor>(params);
            RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                        "Loaded %s extractor (batch=%d): %s",
                        model_name.c_str(), params.batch_size, resolved_engine.c_str());
        }

        // Extractors without native descriptors (RaCo)
        const bool needs_external_descs =
            use_descriptor_matcher_ &&
            feature_extractor_ &&
            feature_extractor_->getDescriptorDim() <= 0 &&
            !desc_head_;
        if (needs_external_descs)
        {
            if (!initDenseAlikedComponents(requested_batch, /*need_dhead=*/true))
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                             "%s + descriptor matcher: failed to initialise dense+dhead "
                             "descriptor source.",
                             model_name.c_str());
            }
        }

        if (use_descriptor_matcher_)
        {
            const std::string matcher_base =
                (std::filesystem::path(weights_folder_).parent_path() / "matcher").string();
            std::string matcher_engine = resolveEnginePath(role,
                                                           {matcher_type_, model_name},
                                                           max_tracked_keypoints_,
                                                           {},
                                                           matcher_base);
            if (matcher_engine.empty())
            {
                RCLCPP_WARN(rclcpp::get_logger("feature_tracker"),
                            "use_descriptor_matcher=true but no '%s' engine found for model='%s', kp=%d "
                            "under '%s/%s'. Descriptor matcher will be disabled.",
                            matcher_type_.c_str(), model_name.c_str(),
                            max_tracked_keypoints_, matcher_base.c_str(), role.c_str());
            }
            else
            {
                uosm::perception::DescriptorMatcherParams mparams;
                mparams.engine_path = matcher_engine;
                mparams.max_keypoints = max_tracked_keypoints_;
                mparams.score_threshold = matcher_score_threshold_;
                // The descriptor dim must match whatever produces descriptors:
                //   dhead (ALIKED dense path or RaCo+matcher), else native extractor.
                if (desc_head_ && desc_head_->is_initialized())
                    mparams.descriptor_dim = desc_head_->getDescriptorDim();
                else if (feature_extractor_ && feature_extractor_->getDescriptorDim() > 0)
                    mparams.descriptor_dim = feature_extractor_->getDescriptorDim();
                else if (feature_extraction_method_ == FeatureExtractionMethod::SIFT_CPU)
                    mparams.descriptor_dim = 128;
                else
                    mparams.descriptor_dim = 256;

                descriptor_matcher_ = uosm::perception::makeDescriptorMatcher(matcher_type_, mparams);
                if (descriptor_matcher_)
                {
                    RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                                "Loaded '%s' descriptor matcher: %s (threshold=%.2f, desc_dim=%d)",
                                descriptor_matcher_->name().c_str(), matcher_engine.c_str(),
                                matcher_score_threshold_, mparams.descriptor_dim);
                }
                else
                {
                    RCLCPP_WARN(rclcpp::get_logger("feature_tracker"),
                                "Unknown matcher_type '%s'; descriptor matcher disabled.",
                                matcher_type_.c_str());
                }
            }
        }
    }

    // Initialize metrics logging with feature extraction method name
    if (metrics_logger_.isEnabled())
    {
        std::string method_name;
        switch (feature_extraction_method_)
        {
        case FeatureExtractionMethod::GFTT_CPU:
            method_name = "gftt_cpu";
            break;
        case FeatureExtractionMethod::GFTT_CUDA:
            method_name = "gftt_cuda";
            break;
        case FeatureExtractionMethod::ALIKED:
            method_name = "aliked";
            break;
        case FeatureExtractionMethod::SuperPoint:
            method_name = "superpoint";
            break;
        case FeatureExtractionMethod::RACO:
            method_name = "raco";
            break;
        case FeatureExtractionMethod::XFeat:
            method_name = "xfeat";
            break;
        case FeatureExtractionMethod::SIFT_CPU:
            method_name = "sift_cpu";
            break;
        }
        metrics_logger_.initFeatureTrackerLog(method_name);
    }
}

FeatureTracker::~FeatureTracker()
{
    if (is_cuda_optflow_init_)
    {
        cleanupCUDAOptflow();
    }
    if (d_prev_kf_desc_)
    {
        cudaFree(d_prev_kf_desc_);
        d_prev_kf_desc_ = nullptr;
        d_prev_kf_desc_capacity_bytes_ = 0;
    }
    if (d_prev_kf_kps_)
    {
        cudaFree(d_prev_kf_kps_);
        d_prev_kf_kps_ = nullptr;
        d_prev_kf_kps_capacity_bytes_ = 0;
    }
    if (d_idx_persistent_)
    {
        cudaFree(d_idx_persistent_);
        d_idx_persistent_ = nullptr;
        d_idx_persistent_capacity_ints_ = 0;
    }
    if (d_sift_desc_)
    {
        cudaFree(d_sift_desc_);
        d_sift_desc_ = nullptr;
        d_sift_desc_cap_ = 0;
    }
    if (d_sift_kps_)
    {
        cudaFree(d_sift_kps_);
        d_sift_kps_ = nullptr;
        d_sift_kps_cap_ = 0;
    }
    if (d_prev_raco_kps_gpu_)
    {
        cudaFree(d_prev_raco_kps_gpu_);
        d_prev_raco_kps_gpu_ = nullptr;
        d_prev_raco_kps_capacity_bytes_ = 0;
    }
    if (d_enc_kpts_n_)
    {
        cudaFree(d_enc_kpts_n_);
        d_enc_kpts_n_ = nullptr;
        d_enc_kpts_n_capacity_bytes_ = 0;
    }
    if (d_enc_image_b2_)
    {
        cudaFree(d_enc_image_b2_);
        d_enc_image_b2_ = nullptr;
        d_enc_image_b2_capacity_bytes_ = 0;
    }

    // ALIKED dense pipeline resources
    if (dl_chain_stream_)
    {
        cudaStreamSynchronize(dl_chain_stream_);
        cudaStreamDestroy(dl_chain_stream_);
        dl_chain_stream_ = nullptr;
    }
    if (d_dkd_workspace_)
    {
        cudaFree(d_dkd_workspace_);
        d_dkd_workspace_ = nullptr;
    }
    if (d_dkd_kpts_px_)
    {
        cudaFree(d_dkd_kpts_px_);
        d_dkd_kpts_px_ = nullptr;
    }
    if (d_dkd_kpts_n_)
    {
        cudaFree(d_dkd_kpts_n_);
        d_dkd_kpts_n_ = nullptr;
    }
    if (d_dkd_scores_)
    {
        cudaFree(d_dkd_scores_);
        d_dkd_scores_ = nullptr;
    }
    if (d_dkd_num_kpts_)
    {
        cudaFree(d_dkd_num_kpts_);
        d_dkd_num_kpts_ = nullptr;
    }
    d_dkd_workspace_bytes_ = 0;
    dkd_buffers_ready_ = false;
}

void FeatureTracker::ensureHostMirrorDesc()
{
    // ALIKED produces descriptors via dense_extractor_+desc_head_, not feature_extractor_,
    // so gate on the GPU buffer state directly. Only left-cam descriptors are mirrored;
    // the right-cam stereo matcher reads cur_dl_d_desc1_ on the GPU directly.
    if (cur_dl_desc_dim_ <= 0 || cur_dl_host_desc0_valid_ ||
        cur_dl_d_desc0_ == nullptr || cur_dl_n0_ <= 0)
        return;

    cur_dl_desc0_.create(cur_dl_n0_, cur_dl_desc_dim_, CV_32F);
    const size_t bytes = static_cast<size_t>(cur_dl_n0_) * cur_dl_desc_dim_ * sizeof(float);
    if (cudaMemcpy(cur_dl_desc0_.data, cur_dl_d_desc0_, bytes, cudaMemcpyDeviceToHost) == cudaSuccess)
        cur_dl_host_desc0_valid_ = true;
    else
        cur_dl_desc0_ = cv::Mat();
}

std::string FeatureTracker::resolveEnginePath(const std::string &role,
                                              const std::vector<std::string> &required_substrings,
                                              int max_kp,
                                              const std::vector<std::string> &forbidden_substrings,
                                              const std::string &base_dir_override) const
{
    const std::string &base = base_dir_override.empty() ? weights_folder_ : base_dir_override;
    if (base.empty())
        return "";

    std::filesystem::path dir(base);
    dir /= role;
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir))
    {
        RCLCPP_WARN(rclcpp::get_logger("feature_tracker"),
                    "resolveEnginePath: directory does not exist: %s", dir.string().c_str());
        return "";
    }

    // ALIKED dense backbone outputs dense feature/score maps and doesn't bake K into its tensor shapes.
    // Ignore kp suffix check.
    const bool kp_relevant = (max_kp >= 0);
    const std::string kp_tag = kp_relevant ? ("_kp" + std::to_string(max_kp)) : std::string();

    auto joinTags = [](const std::vector<std::string> &v)
    {
        std::string out;
        for (size_t i = 0; i < v.size(); ++i)
        {
            out += (i ? "," : "");
            out += "'" + v[i] + "'";
        }
        return out;
    };

    std::vector<std::string> exact_matches;
    std::vector<std::string> loose_matches;

    for (const auto &entry : std::filesystem::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
            continue;
        if (entry.path().extension() != ".engine")
            continue;
        const auto fname = entry.path().filename().string();

        bool all_present = true;
        for (const auto &tag : required_substrings)
        {
            if (!tag.empty() && fname.find(tag) == std::string::npos)
            {
                all_present = false;
                break;
            }
        }
        if (!all_present)
            continue;

        bool any_forbidden = false;
        for (const auto &tag : forbidden_substrings)
        {
            if (!tag.empty() && fname.find(tag) != std::string::npos)
            {
                any_forbidden = true;
                break;
            }
        }
        if (any_forbidden)
            continue;

        if (!kp_relevant || fname.find(kp_tag) != std::string::npos)
            exact_matches.push_back(entry.path().string());
        else
            loose_matches.push_back(entry.path().string());
    }

    if (!exact_matches.empty())
    {
        if (exact_matches.size() > 1)
        {
            RCLCPP_WARN(rclcpp::get_logger("feature_tracker"),
                        "resolveEnginePath: multiple engines match [%s]+%s in %s; picking '%s' "
                        "(consider removing duplicates to disambiguate)",
                        joinTags(required_substrings).c_str(), kp_tag.c_str(),
                        dir.string().c_str(), exact_matches.front().c_str());
        }
        return exact_matches.front();
    }
    if (!loose_matches.empty())
    {
        RCLCPP_WARN(rclcpp::get_logger("feature_tracker"),
                    "resolveEnginePath: no engine matching kp%d for tags [%s] in %s, "
                    "falling back to '%s'",
                    max_kp, joinTags(required_substrings).c_str(),
                    dir.string().c_str(), loose_matches.front().c_str());
        return loose_matches.front();
    }
    return "";
}

void FeatureTracker::initCUDAOptflow()
{
    // Temporal optical flow (frame-to-frame tracking)
    cuda_lk_optflow_ = cv::cuda::SparsePyrLKOpticalFlow::create(
        cv::Size(optflow_window_dim_, optflow_window_dim_),
        optflow_pyramid_levels_,
        optflow_max_iterations_,
        false);

    // Stereo optical flow
    if (use_stereo_mode_)
    {
        cuda_lk_stereo_ = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(stereo_window_dim_, stereo_window_dim_),
            stereo_pyramid_levels_,
            optflow_max_iterations_,
            false);
        RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                    "Stereo optical flow: window=%dx%d, pyramids=%d",
                    stereo_window_dim_, stereo_window_dim_, stereo_pyramid_levels_);
    }

    // Allocate pinned memory buffers for image uploads and GFTT corner downloads
    if (!pinned_buffers_initialized_)
    {
        pinned_img_buffer0_ = cv::cuda::HostMem(row_, col_, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);
        pinned_img_buffer1_ = cv::cuda::HostMem(row_, col_, CV_8UC1, cv::cuda::HostMem::PAGE_LOCKED);
        host_new_pts_buffer_ = cv::cuda::HostMem(1, max_tracked_keypoints_, CV_32FC2, cv::cuda::HostMem::PAGE_LOCKED);
        pinned_buffers_initialized_ = true;
    }

    is_cuda_optflow_init_ = true;
    RCLCPP_DEBUG(rclcpp::get_logger("feature_tracker"), "CUDA optical flow initialized");
}

void FeatureTracker::cleanupCUDAOptflow()
{
    cuda_lk_optflow_.release();
    cuda_lk_stereo_.release();
    prev_gpu_img_.release();
    cur_gpu_img0_.release();
    cur_gpu_img1_.release();
    prev_gpu_pts_.release();
    cur_gpu_pts0_.release();
    cur_gpu_pts1_.release();
    gpu_status_.release();
    gpu_error_.release();
    gpu_new_pts_.release();
    gpu_mask_.release();

    pinned_img_buffer0_.release();
    pinned_img_buffer1_.release();
    host_new_pts_buffer_.release();
    pinned_buffers_initialized_ = false;

    is_cuda_optflow_init_ = false;
    frame_images_uploaded_ = false;
}

int FeatureTracker::getDescriptorDim() const
{
    if (feature_extractor_ && feature_extractor_->is_initialized())
        return feature_extractor_->getDescriptorDim();
    return 0;
}

void FeatureTracker::setStereoBaseline(float baseline)
{
    stereo_baseline_ = baseline;
    RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                "Stereo config: baseline=%.3fm, focal=%.1fpx",
                stereo_baseline_, focal_length_);
}

void FeatureTracker::setStereoExtrinsics(const Eigen::Matrix3d &R_cam0_cam1,
                                         const Eigen::Vector3d &t_cam0_cam1)
{
    R_cam0_cam1_ = R_cam0_cam1;
    t_cam0_cam1_ = t_cam0_cam1;
    stereo_extrinsics_valid_ = true;
    RCLCPP_INFO(rclcpp::get_logger("feature_tracker"),
                "Stereo extrinsics set: t_cam0_cam1=[%.4f, %.4f, %.4f] m (|t|=%.4f)",
                t_cam0_cam1.x(), t_cam0_cam1.y(), t_cam0_cam1.z(),
                t_cam0_cam1.norm());
}

void FeatureTracker::uploadFrameImages()
{
    if (!frame_images_uploaded_)
    {
        // Use pinned memory staging for faster DMA transfers
        if (pinned_buffers_initialized_)
        {
            cv::Mat pinned_mat0 = pinned_img_buffer0_.createMatHeader();
            cur_img0_.copyTo(pinned_mat0);
            cur_gpu_img0_.upload(pinned_img_buffer0_);
            if (!cur_img1_.empty())
            {
                cv::Mat pinned_mat1 = pinned_img_buffer1_.createMatHeader();
                cur_img1_.copyTo(pinned_mat1);
                cur_gpu_img1_.upload(pinned_img_buffer1_);
            }
            else
            {
                cur_gpu_img1_.release();
            }
        }
        else
        {
            // Fallback: direct upload from pageable memory
            cur_gpu_img0_.upload(cur_img0_);
            if (!cur_img1_.empty())
            {
                cur_gpu_img1_.upload(cur_img1_);
            }
            else
            {
                cur_gpu_img1_.release();
            }
        }
        frame_images_uploaded_ = true;
    }
}

void FeatureTracker::trackTemporalCUDA()
{
    if (!frame_images_uploaded_)
    {
        cur_gpu_img0_.upload(cur_img0_);
    }
    if (prev_gpu_img_.empty())
    {
        prev_gpu_img_.upload(prev_img_);
    }

    if (prev_gpu_pts_.empty() || prev_gpu_pts_.cols != static_cast<int>(prev_pts0_.size()))
    {
        cv::Mat prev_pts_mat(1, static_cast<int>(prev_pts0_.size()), CV_32FC2, prev_pts0_.data());
        prev_gpu_pts_.upload(prev_pts_mat);
    }

    // Run GPU optical flow
    cuda_lk_optflow_->calc(prev_gpu_img_, cur_gpu_img0_,
                           prev_gpu_pts_, cur_gpu_pts0_, gpu_status_);

    // Download results
    cv::Mat cur_pts_mat, status_mat;
    cur_gpu_pts0_.download(cur_pts_mat);
    gpu_status_.download(status_mat);

    // Convert to Point2f vector
    cur_pts0_.clear();
    for (int i = 0; i < cur_pts_mat.cols; ++i)
    {
        cur_pts0_.push_back(cur_pts_mat.at<cv::Point2f>(0, i));
    }

    // Convert status to uchar vector
    std::vector<uchar> temporal_status(status_mat.cols);
    for (int i = 0; i < status_mat.cols; ++i)
    {
        temporal_status[i] = status_mat.at<uchar>(0, i);
    }

    // Flow-back check if enabled
    if (allow_flowback_)
    {
        cv::cuda::GpuMat reverse_gpu_pts;
        cv::cuda::GpuMat reverse_status;
        cuda_lk_optflow_->calc(cur_gpu_img0_, prev_gpu_img_,
                               cur_gpu_pts0_, reverse_gpu_pts, reverse_status);

        cv::Mat reverse_pts_mat, reverse_status_mat;
        reverse_gpu_pts.download(reverse_pts_mat);
        reverse_status.download(reverse_status_mat);

        for (size_t i = 0; i < temporal_status.size(); ++i)
        {
            if (temporal_status[i] && reverse_status_mat.at<uchar>(0, static_cast<int>(i)))
            {
                cv::Point2f reverse_pt = reverse_pts_mat.at<cv::Point2f>(0, static_cast<int>(i));
                double dist = distance_sq(prev_pts0_[i], reverse_pt);
                if (dist > OPTFLOW_DISTANCE_THRESHOLD_SQ)
                {
                    temporal_status[i] = 0;
                }
            }
        }
    }

    // Filter border points (and points that drift into the fisheye masked region)
    for (size_t i = 0; i < cur_pts0_.size(); ++i)
    {
        if (!temporal_status[i])
            continue;
        if (!inBorder(cur_pts0_[i]))
            temporal_status[i] = 0;
        else if (isMasked(cur_pts0_[i]))
        {
            temporal_status[i] = 0;
            ++current_metrics_.mask_culled;
        }
    }

    int success_count = std::count(temporal_status.begin(), temporal_status.end(), 1);
    RCLCPP_DEBUG(rclcpp::get_logger("feature_tracker"), "CUDA Forward success: %d / %zu", success_count, temporal_status.size());

    reduceVectors(temporal_status, prev_pts0_, cur_pts0_, ids0_, track_cnt_);

    // Swap GPU images for next frame
    std::swap(prev_gpu_img_, cur_gpu_img0_);
}

void FeatureTracker::trackTemporalLK()
{
    if (optflow_device_ == OpticalFlowBackend::CUDA)
    {
        trackTemporalCUDA();
        return;
    }

    // CPU LK temporal
    std::vector<uchar> status;
    std::vector<float> err;
    const auto win = cv::Size(optflow_window_dim_, optflow_window_dim_);
    const auto term = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                       optflow_max_iterations_, optflow_epsilon_);

    cv::calcOpticalFlowPyrLK(prev_img_, cur_img0_, prev_pts0_, cur_pts0_,
                             status, err, win, optflow_pyramid_levels_);

    if (allow_flowback_)
    {
        Point2fVec rev = prev_pts0_;
        std::vector<uchar> rev_status;
        cv::calcOpticalFlowPyrLK(cur_img0_, prev_img_, cur_pts0_, rev,
                                 rev_status, err, win, 1, term, cv::OPTFLOW_USE_INITIAL_FLOW);
        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i] && rev_status[i])
                status[i] = (distance_sq(prev_pts0_[i], rev[i]) <= OPTFLOW_DISTANCE_THRESHOLD_SQ) ? 1 : 0;
            else
                status[i] = 0;
        }
    }

    for (size_t i = 0; i < cur_pts0_.size(); ++i)
    {
        if (!status[i])
            continue;
        if (!inBorder(cur_pts0_[i]))
            status[i] = 0;
        else if (isMasked(cur_pts0_[i]))
        {
            status[i] = 0;
            ++current_metrics_.mask_culled;
        }
    }

    current_metrics_.forward_total = static_cast<int>(status.size());
    current_metrics_.forward_success = static_cast<int>(std::count(status.begin(), status.end(), 1));
    reduceVectors(status, prev_pts0_, cur_pts0_, ids0_, track_cnt_);
}

void FeatureTracker::trackStereoLK()
{
    cur_pts1_.clear();
    ids1_.clear();
    cur_undistorted_pts1_.clear();
    pts_vel1_.clear();

    if (cur_pts0_.empty())
        return;

    if (optflow_device_ == OpticalFlowBackend::CPU)
    {
        // CPU stereo LK
        std::vector<uchar> status;
        std::vector<float> err;
        const auto win = cv::Size(stereo_window_dim_, stereo_window_dim_);
        const auto term = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                           optflow_max_iterations_, optflow_epsilon_);
        cv::calcOpticalFlowPyrLK(cur_img0_, cur_img1_, cur_pts0_, cur_pts1_,
                                 status, err, win, stereo_pyramid_levels_, term);
        if (allow_flowback_)
        {
            Point2fVec rev;
            std::vector<uchar> rev_status;
            cv::calcOpticalFlowPyrLK(cur_img1_, cur_img0_, cur_pts1_, rev,
                                     rev_status, err, win, stereo_pyramid_levels_, term);
            for (size_t i = 0; i < status.size(); ++i)
            {
                if (status[i] && rev_status[i] && inBorder(cur_pts1_[i]))
                    status[i] = (distance_sq(cur_pts0_[i], rev[i]) <= STEREO_FLOWBACK_THRESHOLD_SQ) ? 1 : 0;
                else
                    status[i] = 0;
            }
        }
        ids1_ = ids0_;
        reduceVectors(status, cur_pts1_, ids1_);
        current_metrics_.stereo_candidates = static_cast<int>(cur_pts0_.size());
        current_metrics_.stereo_success = static_cast<int>(cur_pts1_.size());
        return;
    }

    // CUDA stereo LK
    // TODO: need async cuda? no fused-kernel path here!
    if (!frame_images_uploaded_)
    {
        cur_gpu_img0_.upload(cur_img0_);
        cur_gpu_img1_.upload(cur_img1_);
    }

    cv::Mat cur_pts_mat(1, static_cast<int>(cur_pts0_.size()), CV_32FC2, cur_pts0_.data());
    cur_gpu_pts0_.upload(cur_pts_mat);

    cuda_lk_stereo_->calc(cur_gpu_img0_, cur_gpu_img1_,
                          cur_gpu_pts0_, cur_gpu_pts1_, gpu_status_);

    // Download results
    cv::Mat stereo_pts_mat, stereo_status_mat;
    cur_gpu_pts1_.download(stereo_pts_mat);
    gpu_status_.download(stereo_status_mat);

    // Convert to vectors
    std::vector<uchar> stereo_status(stereo_status_mat.cols);
    cur_pts1_.resize(stereo_pts_mat.cols);
    for (int i = 0; i < stereo_pts_mat.cols; ++i)
    {
        cur_pts1_[i] = stereo_pts_mat.at<cv::Point2f>(0, i);
        stereo_status[i] = stereo_status_mat.at<uchar>(0, i);
    }

    // Flow-back check
    if (allow_flowback_)
    {
        cv::cuda::GpuMat reverse_gpu_pts, reverse_status;
        cuda_lk_stereo_->calc(cur_gpu_img1_, cur_gpu_img0_,
                              cur_gpu_pts1_, reverse_gpu_pts, reverse_status);

        cv::Mat reverse_pts_mat, reverse_status_mat;
        reverse_gpu_pts.download(reverse_pts_mat);
        reverse_status.download(reverse_status_mat);

        for (size_t i = 0; i < stereo_status.size(); ++i)
        {
            if (!stereo_status[i])
                continue;
            if (!reverse_status_mat.at<uchar>(0, static_cast<int>(i)))
            {
                stereo_status[i] = 0;
                continue;
            }
            if (!inBorder(cur_pts1_[i]))
            {
                stereo_status[i] = 0;
                continue;
            }

            cv::Point2f reverse_pt = reverse_pts_mat.at<cv::Point2f>(0, static_cast<int>(i));
            double dist = distance_sq(cur_pts0_[i], reverse_pt);
            if (dist > STEREO_FLOWBACK_THRESHOLD_SQ)
            {
                stereo_status[i] = 0;
            }
        }
    }

    ids1_ = ids0_;
    reduceVectors(stereo_status, cur_pts1_, ids1_);

    current_metrics_.stereo_candidates = static_cast<int>(cur_pts0_.size());
    current_metrics_.stereo_success = static_cast<int>(cur_pts1_.size());
}

void FeatureTracker::recordExtractionTime(
    std::chrono::high_resolution_clock::time_point start)
{
    if (!metrics_logger_.isEnabled())
        return;
    current_metrics_.extraction_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::high_resolution_clock::now() - start)
                                              .count() /
                                          1000.0;
}

void FeatureTracker::featureExtractCUDA()
{
    auto extract_start = std::chrono::high_resolution_clock::now();

    setMask();
    const int n_needed = max_tracked_keypoints_ - static_cast<int>(cur_pts0_.size());
    if (n_needed <= 0)
    {
        n_pts_.clear();
        gpu_new_pts_.release();
        return;
    }

    // Use pre-uploaded image if available, otherwise create temporary GpuMat
    cv::cuda::GpuMat gpu_img;
    if (frame_images_uploaded_ && !cur_gpu_img0_.empty())
    {
        // Image already on GPU - no upload needed
        gpu_img = cur_gpu_img0_;
    }
    else
    {
        // Upload image
        gpu_img.upload(cur_img0_);
    }

    // Upload mask only when content changed
    if (mask_dirty_)
    {
        gpu_mask_.upload(mask_);
        mask_dirty_ = false;
    }

    cuda_gftt_detector_->detect(gpu_img, gpu_new_pts_, gpu_mask_);

    // Download results
    n_pts_.clear();
    if (!gpu_new_pts_.empty())
    {
        gpu_new_pts_.download(host_new_pts_buffer_);
        cv::Mat corners_mat = host_new_pts_buffer_.createMatHeader();

        int n_detected = std::min(n_needed, corners_mat.cols);
        for (int i = 0; i < n_detected; ++i)
        {
            n_pts_.push_back(corners_mat.at<cv::Point2f>(0, i));
        }
    }

    recordExtractionTime(extract_start);

    RCLCPP_DEBUG(rclcpp::get_logger("feature_tracker"), "n_needed: %d, feature size (GFFT CUDA): %zu", n_needed, n_pts_.size());
}

// Convert a raw 2x2 pixel covariance (model space, [c00,c01,c10,c11]) into the residual-space
// square-root information consumed by the reprojection factors
static Eigen::Matrix2d covPxToSqrtInfo(const std::array<float, 4> &cov_px, double focal,
                                       double alpha, double floor_px, double letterbox_scale)
{
    Eigen::Matrix2d Sigma;
    Sigma << cov_px[0], cov_px[1], cov_px[2], cov_px[3];
    Sigma = 0.5 * (Sigma + Sigma.transpose().eval()); // symmetrize
    Sigma *= letterbox_scale * letterbox_scale;       // model px -> original px
    if (floor_px > 0.0)
        Sigma += (floor_px * floor_px) * Eigen::Matrix2d::Identity();
    const double f2 = focal * focal;
    Eigen::Matrix2d info = (f2 * std::max(0.05, alpha)) * Sigma.inverse();
    Eigen::LLT<Eigen::Matrix2d> llt(info);
    if (llt.info() != Eigen::Success || !info.allFinite())
        return (focal / 1.5) * Eigen::Matrix2d::Identity(); // isotropic fallback
    return llt.matrixU();                                   // upper-triangular (sqrt_info)
}

void FeatureTracker::featureExtractDL()
{
    auto extract_start = std::chrono::high_resolution_clock::now();

    if (!feature_extractor_ || !feature_extractor_->is_initialized())
    {
        RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "FeatureExtractor not initialized!");
        n_pts_.clear();
        return;
    }

    // Reset per-frame DL cache (populated below on success).
    cur_dl_valid_ = false;
    cur_dl_stereo_ = false;
    cur_dl_kps0_.clear();
    cur_dl_kps1_.clear();
    cur_dl_desc0_ = cv::Mat();
    cur_dl_d_desc0_ = nullptr;
    cur_dl_d_desc1_ = nullptr;
    cur_dl_n0_ = 0;
    cur_dl_n1_ = 0;
    cur_dl_host_desc0_valid_ = false;
    cur_dl_cov0_.clear();

    auto buildCov = [this](const std::vector<std::array<float, 4>> &raw)
    {
        cur_dl_cov0_.clear();
        if (!feature_extractor_->hasCovariance() || raw.empty())
            return;
        const double focal = focal_length_ > 1.0f ? static_cast<double>(focal_length_) : 460.0;
        const double scale = feature_extractor_->getScale() > 0.0f
                                 ? static_cast<double>(feature_extractor_->getScale())
                                 : 1.0;
        cur_dl_cov0_.reserve(raw.size());
        for (const auto &c : raw)
            cur_dl_cov0_.push_back(covPxToSqrtInfo(c, focal, raco_cov_alpha_,
                                                   raco_cov_floor_px_, scale));
    };

    setMask();
    const int n_needed = max_tracked_keypoints_ - static_cast<int>(cur_pts0_.size());

    std::vector<cv::Point2f> dl_pts0;
    std::vector<float> dl_scores0;
    const bool run_stereo = use_stereo_mode_ && (feature_extractor_->getBatchSize() == 2) &&
                            !cur_img1_.empty();

    if (run_stereo)
    {
        // Upload right image if not already on GPU.
        if (cur_gpu_img1_.empty())
        {
            cur_gpu_img1_.upload(cur_img1_);
        }
        if (cur_gpu_img0_.empty())
        {
            cur_gpu_img0_.upload(cur_img0_);
        }

        if (!feature_extractor_->runInferenceStereoFromGPU(cur_gpu_img0_, cur_gpu_img1_))
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "Stereo GPU inference failed!");
            n_pts_.clear();
            return;
        }
        feature_extractor_->copyOutputsToHostStereo();

        uosm::perception::FeatureExtractor::StereoFeatureData stereo_data;
        feature_extractor_->parseKeypointsStereo(stereo_data);
        buildCov(stereo_data.cov_left);
        dl_pts0 = std::move(stereo_data.keypoints_left);
        dl_scores0 = std::move(stereo_data.scores_left);

        // Cache right-cam (and left-cam) DL outputs for keyframe matcher consumption.
        // GPU pointers are stored without D2H copying; a host mirror is only materialized
        // lazily via ensureHostMirrorDesc() when a consumer (VPR publish / keyframe cache)
        // actually needs CPU-side access.
        cur_dl_kps0_ = dl_pts0;
        cur_dl_kps1_ = std::move(stereo_data.keypoints_right);
        cur_dl_n0_ = static_cast<int>(cur_dl_kps0_.size());
        cur_dl_n1_ = static_cast<int>(cur_dl_kps1_.size());
        cur_dl_desc_dim_ = feature_extractor_->getDescriptorDim();
        if (cur_dl_desc_dim_ > 0)
        {
            if (cur_dl_n0_ > 0)
            {
                cv::cuda::GpuMat desc_left_gpu =
                    feature_extractor_->wrapDescriptorsAsGpuMatStereo(0, cur_dl_n0_);
                cur_dl_d_desc0_ = desc_left_gpu.data;
            }
            if (cur_dl_n1_ > 0)
            {
                cv::cuda::GpuMat desc_right_gpu =
                    feature_extractor_->wrapDescriptorsAsGpuMatStereo(1, cur_dl_n1_);
                cur_dl_d_desc1_ = desc_right_gpu.data;
            }
        }
        cur_dl_stereo_ = true;
        cur_dl_valid_ = true;
    }
    else
    {
        // Mono inference (batch=1): process only left image.
        if (frame_images_uploaded_ && !cur_gpu_img0_.empty())
        {
            if (!feature_extractor_->runInferenceFromGPU(cur_gpu_img0_))
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "GPU inference failed!");
                n_pts_.clear();
                return;
            }
        }
        else
        {
            if (!feature_extractor_->runInferenceSingleImage(cur_img0_))
            {
                RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "Single image inference failed!");
                n_pts_.clear();
                return;
            }
        }

        feature_extractor_->copyOutputsToHostSingle();
        std::vector<std::array<float, 4>> dl_cov0;
        feature_extractor_->parseKeypointsSingle(dl_pts0, dl_scores0, &dl_cov0);
        buildCov(dl_cov0);

        // Cache left-cam DL outputs for keyframe matcher consumption.
        // A GPU pointer is stored without D2H copying; a host mirror is only materialized
        // lazily via ensureHostMirrorDesc() when a consumer (VPR publish / keyframe cache)
        // actually needs CPU-side access.
        cur_dl_kps0_ = dl_pts0;
        cur_dl_n0_ = static_cast<int>(dl_pts0.size());
        cur_dl_desc_dim_ = feature_extractor_->getDescriptorDim();
        if (cur_dl_desc_dim_ > 0 && cur_dl_n0_ > 0)
        {
            cv::cuda::GpuMat desc_left_gpu =
                feature_extractor_->wrapDescriptorsAsGpuMat(cur_dl_n0_);
            cur_dl_d_desc0_ = desc_left_gpu.data;
        }
        cur_dl_stereo_ = false;
        cur_dl_valid_ = true;
    }

    // raco_aliked path: re-describe RACO keypoints with the ALIKED describe engine
    // before any matcher consumes the descriptors.
    runDescHeadHook();

    // Early-out if there's nothing to top up; caches are still populated for keyframe work.
    if (n_needed <= 0)
    {
        n_pts_.clear();
        recordExtractionTime(extract_start);
        return;
    }

    // Filter keypoints by minimum distance from existing tracked points
    n_pts_.clear();
    std::vector<int> accepted_indices;

    for (size_t i = 0; i < dl_pts0.size() && static_cast<int>(n_pts_.size()) < n_needed; ++i)
    {
        const cv::Point2f &candidate = dl_pts0[i];

        // Check bounds
        if (!inBorder(candidate))
            continue;

        // Check mask (if the point falls on masked area)
        int px = cvRound(candidate.x);
        int py = cvRound(candidate.y);
        if (mask_.at<uchar>(py, px) == 0)
            continue;

        // Check distance from existing tracked points
        bool too_close = false;
        for (const auto &existing : cur_pts0_)
        {
            if (distance_sq(candidate, existing) < min_feature_distance_ * min_feature_distance_)
            {
                too_close = true;
                break;
            }
        }

        // Check distance from already added new points
        if (!too_close)
        {
            for (const auto &added : n_pts_)
            {
                if (distance_sq(candidate, added) < min_feature_distance_ * min_feature_distance_)
                {
                    too_close = true;
                    break;
                }
            }
        }

        if (!too_close)
        {
            n_pts_.push_back(candidate);
            accepted_indices.push_back(static_cast<int>(i));
        }
    }

    // Cache descriptors for accepted keypoints (for downstream VPR). This is the one
    // place per frame where we need a host copy of the left-cam descriptors, so lazy-
    // download here rather than at extraction time.
    if (!accepted_indices.empty() && cur_dl_d_desc0_ != nullptr && cur_dl_n0_ > 0)
    {
        ensureHostMirrorDesc();
    }
    if (!accepted_indices.empty() && !cur_dl_desc0_.empty())
    {
        const int desc_dim = cur_dl_desc0_.cols;
        current_descriptors_ = cv::Mat(static_cast<int>(accepted_indices.size()), desc_dim, CV_32F);
        for (size_t j = 0; j < accepted_indices.size(); ++j)
        {
            int src_row = accepted_indices[j];
            if (src_row < cur_dl_desc0_.rows)
            {
                cur_dl_desc0_.row(src_row).copyTo(current_descriptors_.row(static_cast<int>(j)));
            }
        }
    }
    else
    {
        current_descriptors_ = cv::Mat();
    }

    recordExtractionTime(extract_start);

    RCLCPP_DEBUG(rclcpp::get_logger("feature_tracker"), "n_needed: %d, feature size (DL): %zu", n_needed, n_pts_.size());
}

inline bool FeatureTracker::readIntrinsicParameter(const std::string &calib_file, camodocal::CameraPtr &camera)
{
    try
    {
        RCLCPP_DEBUG(rclcpp::get_logger("feature_tracker"), "loading camera model from %s", calib_file.c_str());
        camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    }
    catch (std::exception &e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "Exception: %s", e.what());
        return false;
    }
    return true;
}

inline bool FeatureTracker::inBorder(const cv::Point2f &pt) const
{
    const int img_x = cvRound(pt.x);
    const int img_y = cvRound(pt.y);
    return img_x > 0 && img_x < col_ - 1 &&
           img_y > 0 && img_y < row_ - 1;
}

inline bool FeatureTracker::isMasked(const cv::Point2f &pt) const
{
    return !base_mask_.empty() &&
           base_mask_.at<uchar>(cvRound(pt.y), cvRound(pt.x)) == 0;
}

void FeatureTracker::runDescHeadHook()
{
    if (!cur_dl_valid_ || !feature_extractor_)
        return;
    if (feature_extractor_->getDescriptorDim() > 0)
        return; // extractor already has native descriptors
    if (!use_descriptor_matcher_)
        return; // LK pathway: matcher won't read descriptors
    if (!dense_extractor_ || !dense_extractor_->is_initialized() ||
        !desc_head_ || !desc_head_->is_initialized())
        return;

    const int K = desc_head_->getMaxKp();
    const int B = desc_head_->getBatch();
    const float modelW = static_cast<float>(desc_head_->getModelW());
    const float modelH = static_cast<float>(desc_head_->getModelH());
    if (K <= 0 || (B != 1 && B != 2))
        return;

    // Build kpts_n from the extractor's pixel keypoints, padded to (B, K, 2).
    // The extractor emits original-image pixel coords; map them through the
    // dense extractor's letterbox (model-space normalised [-1, 1]).
    const float scale_letter = (dense_extractor_->getScale() > 0.0f)
                                   ? (1.0f / dense_extractor_->getScale())
                                   : 1.0f;
    const float x_off = dense_extractor_->getXOffset();
    const float y_off = dense_extractor_->getYOffset();
    std::vector<float> kpts_n_host(static_cast<size_t>(B) * K * 2, 0.0f);
    auto fill_slot = [&](int slot, const Point2fVec &kps, int n)
    {
        const size_t base = static_cast<size_t>(slot) * K * 2;
        const int n_use = std::min(n, K);
        for (int i = 0; i < n_use; ++i)
        {
            const float xl = kps[i].x * scale_letter + x_off;
            const float yl = kps[i].y * scale_letter + y_off;
            kpts_n_host[base + i * 2 + 0] = 2.0f * xl / (modelW - 1.0f) - 1.0f;
            kpts_n_host[base + i * 2 + 1] = 2.0f * yl / (modelH - 1.0f) - 1.0f;
        }
    };
    fill_slot(0, cur_dl_kps0_, cur_dl_n0_);
    if (B == 2)
    {
        if (cur_dl_stereo_)
            fill_slot(1, cur_dl_kps1_, cur_dl_n1_);
        else
            fill_slot(1, cur_dl_kps0_, cur_dl_n0_); // duplicate for mono
    }
    const size_t kpts_n_bytes = kpts_n_host.size() * sizeof(float);
    if (kpts_n_bytes > d_enc_kpts_n_capacity_bytes_)
    {
        if (d_enc_kpts_n_)
            cudaFree(d_enc_kpts_n_);
        if (cudaMalloc(reinterpret_cast<void **>(&d_enc_kpts_n_), kpts_n_bytes) != cudaSuccess)
        {
            d_enc_kpts_n_ = nullptr;
            d_enc_kpts_n_capacity_bytes_ = 0;
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                         "dhead kpts_n cudaMalloc(%zu) failed", kpts_n_bytes);
            return;
        }
        d_enc_kpts_n_capacity_bytes_ = kpts_n_bytes;
    }
    if (cudaMemcpyAsync(d_enc_kpts_n_, kpts_n_host.data(), kpts_n_bytes,
                        cudaMemcpyHostToDevice, dl_chain_stream_) != cudaSuccess)
    {
        RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "dhead kpts_n upload failed");
        return;
    }

    // Run dense extractor on raw GPU images (same images the extractor consumed).
    bool dense_ok = false;
    if (cur_dl_stereo_ && B == 2 && !cur_gpu_img0_.empty() && !cur_gpu_img1_.empty())
        dense_ok = dense_extractor_->runInferenceStereoFromGPU(cur_gpu_img0_, cur_gpu_img1_, dl_chain_stream_);
    else if (B == 1 && !cur_gpu_img0_.empty())
        dense_ok = dense_extractor_->runInferenceFromGPU(cur_gpu_img0_, dl_chain_stream_);
    else if (B == 2 && !cur_gpu_img0_.empty())
        // Mono extractor with a B=2 dense engine: feed the same image into both slots.
        dense_ok = dense_extractor_->runInferenceStereoFromGPU(cur_gpu_img0_, cur_gpu_img0_, dl_chain_stream_);
    if (!dense_ok)
    {
        RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "dhead: dense extractor failed");
        return;
    }

    if (!desc_head_->runInference(dense_extractor_->getFeatureMapGpu(),
                                  d_enc_kpts_n_, dl_chain_stream_))
    {
        RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "dhead inference failed");
        return;
    }
    cudaStreamSynchronize(dl_chain_stream_);

    float *d_desc = static_cast<float *>(desc_head_->getDescriptorsGpu());
    const int D = desc_head_->getDescriptorDim();
    cur_dl_desc_dim_ = D;
    cur_dl_d_desc0_ = d_desc;
    cur_dl_d_desc1_ = (cur_dl_stereo_ && B == 2)
                          ? d_desc + static_cast<size_t>(K) * D
                          : nullptr;
    cur_dl_host_desc0_valid_ = false;
}

ObservationsMap FeatureTracker::trackImage(double _cur_time,
                                           const cv::Mat &_img0,
                                           const cv::Mat &_img1)
{
    auto global_start = std::chrono::high_resolution_clock::now();

    // Reset per-frame metrics
    current_metrics_ = FrameMetrics();
    current_metrics_.frame_id = frame_counter_++;
    current_metrics_.timestamp = _cur_time;

    stereo_match_viz_.clear();

    if (_img0.empty())
    {
        throw std::runtime_error("No image provided");
        return ObservationsMap();
    }

    cur_time_ = _cur_time;
    cur_img0_ = _img0.clone();
    if (!_img1.empty())
        cur_img1_ = _img1.clone();
    else
        cur_img1_ = cv::Mat();

    // Clear descriptor cache before tracking
    descriptor_ids_.clear();
    dl_new_left_ids_.clear();
    cur_match_cov_.clear();

    // ── Core tracking ─────────────────────────────────────────────────────────
    const bool dl_active = dlDescriptorTemporalActive();

    if (dl_active)
    {
        // Branch A: DL + LightGlue (SuperPoint / ALIKED / RaCo / XFeat / SIFT_CPU)
        uploadFrameImages();
        if (feature_extraction_method_ == FeatureExtractionMethod::ALIKED)
            featureExtractAliked();
        else if (feature_extraction_method_ == FeatureExtractionMethod::SIFT_CPU)
            featureExtractSIFT();
        else
            featureExtractDL();
        matchTemporalDescriptors();
    }
    else
    {
        // Branch B: GFTT (CPU or CUDA) or DL+LK
        cur_pts0_.clear();
        if (!prev_pts0_.empty())
            trackTemporalLK();

        switch (feature_extraction_method_)
        {
        case FeatureExtractionMethod::GFTT_CPU:
            featureExtractOpenCV();
            break;
        case FeatureExtractionMethod::GFTT_CUDA:
            featureExtractCUDA();
            break;
        case FeatureExtractionMethod::ALIKED:
            featureExtractAliked();
            break;
        case FeatureExtractionMethod::SIFT_CPU:
            featureExtractSIFT();
            break;
        default: // SuperPoint, RACO, XFeat (+LK)
            featureExtractDL();
            break;
        }

        current_metrics_.new_features = static_cast<int>(n_pts_.size());
        for (const auto &p : n_pts_)
        {
            cur_pts0_.push_back(p);
            ids0_.push_back(n_id_++);
            track_cnt_.push_back(1);
        }

        if (!use_descriptor_matcher_ && !cur_pts0_.empty() && !cur_img1_.empty())
            trackStereoLK();
    }

    // ── Undistort + velocities ─────────────────────────────────────────────────
    cur_undistorted_pts0_ = undistortPoints(cur_pts0_, m_cameras_[0]);
    pts_vel0_ = calcVelfromPoints(ids0_, cur_undistorted_pts0_, prev_undistorted_pts0_map_);
    if (m_cameras_[1])
    {
        cur_undistorted_pts1_ = undistortPoints(cur_pts1_, m_cameras_[1]);
        pts_vel1_ = calcVelfromPoints(ids1_, cur_undistorted_pts1_, prev_undistorted_pts1_map_);
    }
    else
    {
        cur_pts1_.clear();
        ids1_.clear();
        cur_undistorted_pts1_.clear();
        pts_vel1_.clear();
    }

    // ── Next-frame state ───────────────────────────────────────────────────────
    prev_img_ = cur_img0_;
    prev_pts0_ = cur_pts0_;
    prev_time_ = cur_time_;
    if (!dl_active && optflow_device_ == OpticalFlowBackend::CUDA)
    {
        // trackTemporalCUDA() already swapped prev_gpu_img_/cur_gpu_img0_.
        // Upload filtered tracking points for the next frame.
        if (!cur_pts0_.empty())
        {
            cv::Mat m(1, static_cast<int>(cur_pts0_.size()), CV_32FC2, cur_pts0_.data());
            prev_gpu_pts_.upload(m);
        }
        else
            prev_gpu_pts_.release();
    }
    frame_images_uploaded_ = false;

    // ── Descriptor export (loop closure) ──────────────────────────────────────
    if (!dl_new_left_ids_.empty() && cur_dl_d_desc0_ != nullptr &&
        cur_dl_n0_ > 0 && cur_dl_desc_dim_ > 0)
    {
        ensureHostMirrorDesc();
        if (!cur_dl_desc0_.empty())
        {
            std::vector<int> new_idx;
            new_idx.reserve(dl_new_left_ids_.size());
            for (size_t i = 0; i < dl_new_left_ids_.size(); ++i)
                if (dl_new_left_ids_[i] >= 0)
                    new_idx.push_back(static_cast<int>(i));
            if (!new_idx.empty())
            {
                current_descriptors_ = cv::Mat(static_cast<int>(new_idx.size()),
                                               cur_dl_desc_dim_, CV_32F);
                descriptor_ids_.resize(new_idx.size());
                for (size_t j = 0; j < new_idx.size(); ++j)
                {
                    int i = new_idx[j];
                    cur_dl_desc0_.row(i).copyTo(current_descriptors_.row(static_cast<int>(j)));
                    descriptor_ids_[j] = dl_new_left_ids_[i];
                }
            }
            else
            {
                current_descriptors_ = cv::Mat();
            }
        }
    }

    // ── Visualization ──────────────────────────────────────────────────────────
    if (enable_visualization_)
    {
        if (!use_descriptor_matcher_)
            drawTrack();
        prev_pts0_viz_ = prev_pts0_map;
        prev_pts0_map.clear();
        for (size_t i = 0; i < cur_pts0_.size(); ++i)
            prev_pts0_map[ids0_[i]] = cur_pts0_[i];
    }

    // DL+LG stereo

    // ── Build ObservationsMap ──────────────────────────────────────────────────
    ObservationsMap observations;
    for (size_t i = 0; i < ids0_.size(); ++i)
    {
        int feature_id = ids0_[i];
        Observation obs;
        obs.point_c = {cur_undistorted_pts0_[i].x, cur_undistorted_pts0_[i].y, 1.0};
        obs.uv = {cur_pts0_[i].x, cur_pts0_[i].y};
        obs.velocity = {pts_vel0_[i].x, pts_vel0_[i].y};
        const auto cov_it = cur_match_cov_.find(feature_id);
        if (cov_it != cur_match_cov_.end())
        {
            obs.has_cov = true;
            obs.sqrt_info_px = cov_it->second;
        }
        observations[feature_id].emplace_back(0, obs);
    }

    const size_t right_count = std::min(
        std::min(ids1_.size(), cur_undistorted_pts1_.size()),
        std::min(cur_pts1_.size(), pts_vel1_.size()));
    for (size_t i = 0; i < right_count; ++i)
    {
        if (use_descriptor_matcher_ &&
            observations.find(ids1_[i]) == observations.end())
            continue;
        int feature_id = ids1_[i];
        Observation obs;
        obs.point_c = {cur_undistorted_pts1_[i].x, cur_undistorted_pts1_[i].y, 1.0};
        obs.uv = {cur_pts1_[i].x, cur_pts1_[i].y};
        obs.velocity = {pts_vel1_[i].x, pts_vel1_[i].y};
        observations[feature_id].emplace_back(1, obs);
    }

    // ── Keyframe cache update ──────────────────────────────────────────────────
    if (use_descriptor_matcher_)
        updateKeyframeCache();

    // ── Metrics ────────────────────────────────────────────────────────────────
    if (metrics_logger_.isEnabled())
    {
        auto global_end = std::chrono::high_resolution_clock::now();
        current_metrics_.total_features = static_cast<int>(cur_pts0_.size());
        current_metrics_.total_time_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(global_end - global_start)
                .count() /
            1000.0;
        metrics_logger_.logFeatureTracker(current_metrics_);
    }

    return observations;
}

void FeatureTracker::featureExtractOpenCV()
{
    auto extract_start = std::chrono::high_resolution_clock::now();
    setMask();
    const int n_needed = max_tracked_keypoints_ - static_cast<int>(cur_pts0_.size());
    if (n_needed > 0)
    {
        if (mask_.empty())
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "mask is empty!");
        if (mask_.type() != CV_8UC1)
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "mask type wrong!");

        cv::goodFeaturesToTrack(cur_img0_, n_pts_, n_needed, 0.01, min_feature_distance_, mask_);
    }
    else
    {
        n_pts_.clear();
    }
    RCLCPP_DEBUG(rclcpp::get_logger("feature_tracker"), "n_needed: %d, feature size (GFFT): %zu", n_needed, n_pts_.size());

    recordExtractionTime(extract_start);
    current_metrics_.new_features = static_cast<int>(n_pts_.size());
}

void FeatureTracker::featureExtractSIFT()
{
    auto extract_start = std::chrono::high_resolution_clock::now();

    // Reset per-frame DL cache (mirrors the reset in featureExtractDL)
    cur_dl_valid_ = false;
    cur_dl_stereo_ = false;
    cur_dl_kps0_.clear();
    cur_dl_kps1_.clear();
    cur_dl_d_desc0_ = nullptr;
    cur_dl_d_desc1_ = nullptr;
    cur_dl_n0_ = 0;
    cur_dl_n1_ = 0;
    cur_dl_host_desc0_valid_ = false;
    cur_dl_desc_dim_ = 0;

    constexpr int D = 128;
    const int K = max_tracked_keypoints_;

    if (sift_detector_.empty())
        sift_detector_ = cv::SIFT::create(K);

    // Normalize each descriptor row to unit L2 (matches sift_lightglue.pth convention)
    auto normalizeRows = [D](cv::Mat &m)
    {
        for (int i = 0; i < m.rows; ++i)
        {
            float *row = m.ptr<float>(i);
            float sq = 0.0f;
            for (int j = 0; j < D; ++j)
                sq += row[j] * row[j];
            const float inv = 1.0f / (std::sqrt(sq) + 1e-8f);
            for (int j = 0; j < D; ++j)
                row[j] *= inv;
        }
    };

    // Pack keypoints as (K, 4) float32: x_px, y_px, scale, ori_rad
    auto uploadKps = [&](int slot, const std::vector<cv::KeyPoint> &kps)
    {
        std::vector<float> tmp(static_cast<size_t>(K) * 4, 0.0f);
        const int n = static_cast<int>(kps.size());
        for (int i = 0; i < n; ++i)
        {
            tmp[i * 4 + 0] = kps[i].pt.x;
            tmp[i * 4 + 1] = kps[i].pt.y;
            tmp[i * 4 + 2] = kps[i].size;
            tmp[i * 4 + 3] = kps[i].angle * static_cast<float>(CV_PI / 180.0);
        }
        float *dst = d_sift_kps_ + static_cast<size_t>(slot) * K * 4;
        cudaMemcpy(dst, tmp.data(), static_cast<size_t>(K) * 4 * sizeof(float),
                   cudaMemcpyHostToDevice);
    };

    // Upload N×D descriptor matrix to GPU slot (zero-padded to K rows for TRT)
    auto uploadDescs = [&](int slot, const cv::Mat &descs, int n)
    {
        const int n_use = std::min(n, K);
        float *dst = d_sift_desc_ + static_cast<size_t>(slot) * K * D;
        const size_t slot_bytes = static_cast<size_t>(K) * D * sizeof(float);
        cudaMemset(dst, 0, slot_bytes);
        if (n_use > 0 && !descs.empty())
        {
            cudaMemcpy(dst, descs.ptr<float>(0),
                       static_cast<size_t>(n_use) * D * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
    };

    // --- Left camera ---
    std::vector<cv::KeyPoint> kps0;
    cv::Mat desc0;
    if (use_descriptor_matcher_)
        sift_detector_->detectAndCompute(cur_img0_, cv::noArray(), kps0, desc0);
    else
        sift_detector_->detect(cur_img0_, kps0);

    if (static_cast<int>(kps0.size()) > K)
        kps0.resize(static_cast<size_t>(K));
    cur_dl_kps0_.reserve(kps0.size());
    for (const auto &kp : kps0)
        cur_dl_kps0_.emplace_back(kp.pt);
    cur_dl_n0_ = static_cast<int>(cur_dl_kps0_.size());

    if (use_descriptor_matcher_ && cur_dl_n0_ > 0 && !desc0.empty())
    {
        if (desc0.rows > cur_dl_n0_)
            desc0 = desc0.rowRange(0, cur_dl_n0_);
        if (desc0.type() != CV_32F)
            desc0.convertTo(desc0, CV_32F);
        normalizeRows(desc0);
        uploadKps(0, kps0);
        uploadDescs(0, desc0, cur_dl_n0_);
        cur_dl_d_desc0_ = d_sift_desc_;
        cur_dl_desc_dim_ = D;
    }

    // --- Right camera (stereo) ---
    if (use_stereo_mode_ && !cur_img1_.empty())
    {
        std::vector<cv::KeyPoint> kps1;
        cv::Mat desc1;
        if (use_descriptor_matcher_)
            sift_detector_->detectAndCompute(cur_img1_, cv::noArray(), kps1, desc1);
        else
            sift_detector_->detect(cur_img1_, kps1);

        if (static_cast<int>(kps1.size()) > K)
            kps1.resize(static_cast<size_t>(K));
        cur_dl_kps1_.reserve(kps1.size());
        for (const auto &kp : kps1)
            cur_dl_kps1_.emplace_back(kp.pt);
        cur_dl_n1_ = static_cast<int>(cur_dl_kps1_.size());

        if (use_descriptor_matcher_ && cur_dl_n1_ > 0 && !desc1.empty())
        {
            if (desc1.rows > cur_dl_n1_)
                desc1 = desc1.rowRange(0, cur_dl_n1_);
            if (desc1.type() != CV_32F)
                desc1.convertTo(desc1, CV_32F);
            normalizeRows(desc1);
            uploadKps(1, kps1);
            uploadDescs(1, desc1, cur_dl_n1_);
            cur_dl_d_desc1_ = d_sift_desc_ + static_cast<size_t>(K) * D;
        }
        cur_dl_stereo_ = true;
    }

    cur_dl_valid_ = true;

    // For the LK path (use_descriptor_matcher_=false): populate n_pts_ with
    // SIFT keypoints that are not already covered by tracked points
    if (!use_descriptor_matcher_)
    {
        setMask();
        n_pts_.clear();
        const int n_needed = K - static_cast<int>(cur_pts0_.size());
        for (const auto &kp : kps0)
        {
            if (static_cast<int>(n_pts_.size()) >= n_needed)
                break;
            if (!inBorder(kp.pt))
                continue;
            if (mask_.at<uchar>(kp.pt) == 255)
            {
                n_pts_.push_back(kp.pt);
                cv::circle(mask_, kp.pt, min_feature_distance_, 0, -1);
            }
        }
        current_metrics_.new_features = static_cast<int>(n_pts_.size());
    }
    else
    {
        current_metrics_.new_features = cur_dl_n0_;
    }

    recordExtractionTime(extract_start);
}

void FeatureTracker::setMask()
{
    if (!base_mask_.empty())
        base_mask_.copyTo(mask_);
    else if (mask_.rows != row_ || mask_.cols != col_ || mask_.type() != CV_8UC1)
        mask_ = cv::Mat(row_, col_, CV_8UC1, cv::Scalar(255));
    else
        mask_.setTo(cv::Scalar(255));
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> sorted_pts;
    for (size_t i = 0; i < cur_pts0_.size(); ++i)
    {
        sorted_pts.push_back(std::make_pair(track_cnt_[i], std::make_pair(cur_pts0_[i], ids0_[i])));
    }

    std::sort(sorted_pts.begin(), sorted_pts.end(),
              [](const std::pair<int, std::pair<cv::Point2f, int>> &a,
                 const std::pair<int, std::pair<cv::Point2f, int>> &b)
              {
                  return a.first > b.first;
              });
    cur_pts0_.clear();
    ids0_.clear();
    track_cnt_.clear();
    for (const auto &it : sorted_pts)
    {
        if (mask_.at<uchar>(it.second.first) == 255)
        {
            cur_pts0_.push_back(it.second.first);
            ids0_.push_back(it.second.second);
            track_cnt_.push_back(it.first);
            // update mask
            cv::circle(mask_, it.second.first, min_feature_distance_, 0, -1);
        }
    }
    // Mark mask as dirty for GPU upload
    mask_dirty_ = true;
}

Point2fVec FeatureTracker::undistortPoints(Point2fVec &pts, camodocal::CameraPtr &cam)
{
    Point2fVec undistorted_pts;
    undistorted_pts.reserve(pts.size());
    for (const auto &pt : pts)
    {
        Eigen::Vector2d a(pt.x, pt.y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        undistorted_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
    return undistorted_pts;
}

Point2fVec FeatureTracker::calcVelfromPoints(const std::vector<int> &ids, const Point2fVec &pts,
                                             Point2fMap &prev_id_pts)
{
    Point2fVec pts_velocity;
    pts_velocity.reserve(ids.size());
    const double dt = cur_time_ - prev_time_;

    for (size_t i = 0; i < ids.size(); ++i)
    {
        auto it = prev_id_pts.find(ids[i]);
        if (it != prev_id_pts.end())
        {
            const double v_x = (pts[i].x - it->second.x) / dt;
            const double v_y = (pts[i].y - it->second.y) / dt;
            pts_velocity.emplace_back(v_x, v_y);
        }
        else
        {
            pts_velocity.emplace_back(0, 0);
        }
    }
    // Clear and build map for next frame
    prev_id_pts.clear();
    for (size_t i = 0; i < ids.size(); ++i)
    {
        prev_id_pts.insert(std::make_pair(ids[i], pts[i]));
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack()
{
    if (cur_img0_.empty())
    {
        RCLCPP_WARN(rclcpp::get_logger("feature_tracker"), "Cannot draw tracking visualization, image is empty");
        return;
    }

    uosm::utility::DebugVisualizer::LkFrameInputs in{
        cur_img0_, cur_img1_,
        cur_pts0_, cur_pts1_,
        ids0_, track_cnt_,
        prev_pts0_map};
    uosm::utility::DebugVisualizer::LkStats stats{
        static_cast<int>(cur_pts0_.size()),
        current_metrics_.new_features,
        static_cast<int>(cur_pts1_.size()),
        static_cast<int>(cur_pts0_.size()),
        current_metrics_.mask_culled,
        !base_mask_.empty()};
    tracked_image_ = uosm::utility::DebugVisualizer::renderLkFrame(in, stats);
}

cv::Mat FeatureTracker::getTrackVisualization()
{
    if (use_descriptor_matcher_)
    {
        cv::Mat viz = getDescriptorDebugViz();
        if (!viz.empty())
            return viz;
    }
    return tracked_image_;
}

cv::Mat FeatureTracker::getDescriptorDebugViz()
{
    if (!use_descriptor_matcher_)
        return cv::Mat();
    if (cur_img0_.empty())
        return cv::Mat();

    std::vector<uosm::utility::DebugVisualizer::StereoMatch> stereo_matches;
    stereo_matches.reserve(stereo_match_viz_.size());
    for (const auto &t : stereo_match_viz_)
        stereo_matches.push_back({t.idx_L, t.idx_R, t.outcome});

    uosm::utility::DebugVisualizer::DescriptorFrameInputs in{
        cur_img0_, cur_img1_,
        cur_pts0_, ids0_, track_cnt_, prev_pts0_viz_,
        cur_dl_kps0_, cur_dl_kps1_,
        stereo_matches};

    const auto &m = current_metrics_;
    uosm::utility::DebugVisualizer::DescriptorStats stats;
    stats.tracked = static_cast<int>(cur_pts0_.size());
    stats.stereo_candidates = m.stereo_candidates;
    stats.stereo_admitted = m.stereo_success;
    stats.stereo_rej_depth = m.stereo_gate_depth_rejects;
    stats.stereo_rej_reproj = m.stereo_gate_reproj_rejects;
    stats.temporal_matches = m.forward_success;
    stats.matcher_ms = m.matcher_time_ms;
    stats.frame_idx = frame_counter_;
    stats.mask_culled = m.mask_culled;
    stats.fisheye = !base_mask_.empty();

    return uosm::utility::DebugVisualizer::renderDescriptorFrame(in, stats);
}

bool FeatureTracker::dlDescriptorTemporalActive() const
{
    return use_descriptor_matcher_ && descriptor_matcher_ &&
           descriptor_matcher_->is_initialized();
}

void FeatureTracker::applyTemporalMatches(
    const std::vector<uosm::perception::DescriptorMatch> &matches,
    const std::vector<int> &prev_ids,
    const std::vector<int> &prev_track_cnt,
    int n_cur,
    std::vector<int> &cur_assigned_id,
    std::vector<int> &cur_track_cnt,
    std::unordered_set<int> &consumed_prev) const
{
    const int n_prev = static_cast<int>(prev_ids.size());
    consumed_prev.reserve(matches.size());
    for (const auto &m : matches)
    {
        if (m.idx0 < 0 || m.idx0 >= n_prev)
            continue;
        if (m.idx1 < 0 || m.idx1 >= n_cur)
            continue;
        const int prev_id = prev_ids[m.idx0];
        if (prev_id < 0)
            continue;
        if (consumed_prev.count(prev_id))
            continue;
        if (cur_assigned_id[m.idx1] != -1)
            continue;

        cur_assigned_id[m.idx1] = prev_id;
        cur_track_cnt[m.idx1] =
            (m.idx0 < static_cast<int>(prev_track_cnt.size())
                 ? prev_track_cnt[m.idx0]
                 : 1) +
            1;
        consumed_prev.insert(prev_id);
    }
}

int FeatureTracker::accumulateLeftOutputs(
    const std::vector<cv::Point2f> &cur_dl_kps,
    const std::vector<int> &cur_assigned_id,
    const std::vector<int> &cur_track_cnt)
{
    const int n_cur = static_cast<int>(cur_dl_kps.size());
    dl_to_pts0_idx_.assign(n_cur, -1);
    dl_new_left_ids_.assign(n_cur, -1);
    int inherited_count = 0;
    for (int i = 0; i < n_cur; ++i)
    {
        const cv::Point2f &kp = cur_dl_kps[i];
        if (!inBorder(kp))
            continue;
        if (isMasked(kp))
        {
            ++current_metrics_.mask_culled;
            continue;
        }
        dl_to_pts0_idx_[i] = static_cast<int>(cur_pts0_.size());
        cur_pts0_.push_back(kp);
        const bool inherited = (cur_assigned_id[i] >= 0);
        const int assigned_id = inherited ? cur_assigned_id[i] : n_id_++;
        ids0_.push_back(assigned_id);
        track_cnt_.push_back(cur_track_cnt[i]);
        if (inherited)
            ++inherited_count;
        else
            dl_new_left_ids_[i] = assigned_id;
        if (i < static_cast<int>(cur_dl_cov0_.size()))
            cur_match_cov_[assigned_id] = cur_dl_cov0_[i];
    }
    return inherited_count;
}

void FeatureTracker::seedLeftFromCurrent(const std::vector<cv::Point2f> &cur_dl_kps)
{
    const int n = static_cast<int>(cur_dl_kps.size());
    dl_to_pts0_idx_.assign(n, -1);
    dl_new_left_ids_.assign(n, -1);
    for (int i = 0; i < n; ++i)
    {
        const cv::Point2f &kp = cur_dl_kps[i];
        if (!inBorder(kp))
            continue;
        if (isMasked(kp))
        {
            ++current_metrics_.mask_culled;
            continue;
        }
        dl_to_pts0_idx_[i] = static_cast<int>(cur_pts0_.size());
        cur_pts0_.push_back(kp);
        const int assigned_id = n_id_++;
        ids0_.push_back(assigned_id);
        track_cnt_.push_back(1);
        dl_new_left_ids_[i] = assigned_id;
        if (i < static_cast<int>(cur_dl_cov0_.size()))
            cur_match_cov_[assigned_id] = cur_dl_cov0_[i];
    }
}

int FeatureTracker::filterMatchesE(
    std::vector<uosm::perception::DescriptorMatch> &matches,
    const Point2fVec &prev_kps,
    const Point2fVec &cur_kps,
    const camodocal::CameraPtr &cam) const
{
    // Undistort both sides with one model.
    return filterMatchesE(matches, prev_kps, cur_kps, cam, cam);
}

int FeatureTracker::filterMatchesE(
    std::vector<uosm::perception::DescriptorMatch> &matches,
    const Point2fVec &kps0,
    const Point2fVec &kps1,
    const camodocal::CameraPtr &cam0,
    const camodocal::CameraPtr &cam1) const
{
    constexpr float kEpipolarThreshPx = 1.0f; // E-matrix RANSAC inlier threshold
    if (!cam0 || !cam1 || focal_length_ <= 0.0f)
        return 0;
    if (matches.size() < 8) // E-matrix minimum correspondences
        return 0;

    Point2fVec pts_prev_px, pts_cur_px;
    pts_prev_px.reserve(matches.size());
    pts_cur_px.reserve(matches.size());
    std::vector<int> kept_idx;
    kept_idx.reserve(matches.size());
    const int n_prev = static_cast<int>(kps0.size());
    const int n_cur = static_cast<int>(kps1.size());
    for (int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        const auto &m = matches[i];
        if (m.idx0 < 0 || m.idx0 >= n_prev || m.idx1 < 0 || m.idx1 >= n_cur)
            continue;
        pts_prev_px.push_back(kps0[m.idx0]);
        pts_cur_px.push_back(kps1[m.idx1]);
        kept_idx.push_back(i);
    }
    if (pts_prev_px.size() < 8)
        return 0;

    camodocal::CameraPtr cam0_ = cam0;
    camodocal::CameraPtr cam1_ = cam1;
    Point2fVec pts_prev_n = const_cast<FeatureTracker *>(this)->undistortPoints(pts_prev_px, cam0_);
    Point2fVec pts_cur_n = const_cast<FeatureTracker *>(this)->undistortPoints(pts_cur_px, cam1_);
    if (pts_prev_n.size() != pts_cur_n.size() || pts_prev_n.size() < 8)
        return 0;

    const double threshold_n = static_cast<double>(kEpipolarThreshPx) /
                               static_cast<double>(focal_length_);
    constexpr int kEpipolarRansacMaxIters = 200;
    const cv::Mat K_identity = cv::Mat::eye(3, 3, CV_64F);
    std::vector<uchar> inlier_mask;
    cv::findEssentialMat(pts_prev_n, pts_cur_n, K_identity,
                         cv::USAC_MAGSAC, 0.999, threshold_n,
                         kEpipolarRansacMaxIters, inlier_mask);
    if (inlier_mask.size() != pts_prev_n.size())
        return 0; // RANSAC failed; keep matches as-is

    // Guard against degenerate motion (pure rotation, near-zero baseline) as
    // the E-matrix cant do so.
    const int n_in = std::count(inlier_mask.begin(), inlier_mask.end(), 1);
    if (static_cast<double>(n_in) < 0.5 * static_cast<double>(inlier_mask.size()))
        return 0;

    std::vector<uosm::perception::DescriptorMatch> filtered;
    filtered.reserve(kept_idx.size());
    int rejected = 0;
    for (size_t k = 0; k < kept_idx.size(); ++k)
    {
        if (inlier_mask[k])
            filtered.push_back(matches[kept_idx[k]]);
        else
            ++rejected;
    }
    matches = std::move(filtered);
    return rejected;
}

void FeatureTracker::trackTemporalDescriptorLeft()
{
    auto t0 = std::chrono::high_resolution_clock::now();

    cur_pts0_.clear();
    ids0_.clear();
    track_cnt_.clear();

    if (!cur_dl_valid_ || cur_dl_kps0_.empty() || cur_dl_d_desc0_ == nullptr || cur_dl_n0_ <= 0)
        return;

    const int n_cur = cur_dl_n0_;

    // First-frame (or no prev cache): adopt all DL kps as fresh tracks.
    if (!prev_kf_valid_ || prev_kf_left_kps_.empty() || d_prev_kf_desc_ == nullptr)
    {
        seedLeftFromCurrent(cur_dl_kps0_);
        current_metrics_.forward_total = 0;
        current_metrics_.forward_success = 0;
        current_metrics_.matcher_time_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t0)
                .count() /
            1000.0;
        return;
    }

    const int n_prev = static_cast<int>(prev_kf_left_kps_.size());
    const float *cur_kps_gpu = currentKeypointsGpuBatch(0);
    const auto t_infer = std::chrono::high_resolution_clock::now();
    auto matches = descriptor_matcher_->match(d_prev_kf_kps_, n_prev, d_prev_kf_desc_,
                                              cur_kps_gpu, n_cur, cur_dl_d_desc0_,
                                              row_, col_);
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t_infer)
            .count() /
        1000.0;
    const auto t_ransac = std::chrono::high_resolution_clock::now();
    current_metrics_.ransac_rejected_temporal +=
        filterMatchesE(matches, prev_kf_left_kps_, cur_dl_kps0_, m_cameras_[0]);
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t_ransac)
            .count() /
        1000.0;

    std::vector<int> cur_assigned_id(n_cur, -1);
    std::vector<int> cur_track_cnt(n_cur, 1);
    std::unordered_set<int> consumed_prev;
    applyTemporalMatches(matches, prev_kf_left_ids_, prev_kf_left_track_cnt_,
                         n_cur, cur_assigned_id, cur_track_cnt, consumed_prev);

    accumulateLeftOutputs(cur_dl_kps0_, cur_assigned_id, cur_track_cnt);

    current_metrics_.matcher_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0)
            .count() /
        1000.0;
    current_metrics_.forward_total = n_prev;
    current_metrics_.forward_success = static_cast<int>(consumed_prev.size());
}

void FeatureTracker::matchTemporalDescriptors()
{
    stereo_matches_pending_ = false;

    const bool stereo_active = use_stereo_mode_ && cur_dl_stereo_;

    // Mono DL: single left temporal match via non-batched match() (B=1, mono engine).
    if (!stereo_active)
    {
        trackTemporalDescriptorLeft();
        return;
    }

    const bool left_has_prev = prev_kf_valid_ && !prev_kf_left_kps_.empty() && d_prev_kf_desc_ && d_prev_kf_kps_;

    if (!cur_dl_valid_ || cur_dl_kps0_.empty() || cur_dl_kps1_.empty() ||
        cur_dl_d_desc0_ == nullptr || cur_dl_d_desc1_ == nullptr ||
        cur_dl_n0_ <= 0 || cur_dl_n1_ <= 0)
        return;

    // No prev cache (first frame or extractor dropout): seed left fresh, skip matcher.
    // The right camera is a per-frame stereo anchor only — without a left track there
    // is nothing to anchor, so it is left empty for this frame.
    if (!left_has_prev)
    {
        seedLeftFromCurrent(cur_dl_kps0_);
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    cur_pts0_.clear();
    ids0_.clear();
    track_cnt_.clear();
    cur_pts1_.clear();
    ids1_.clear();
    track_cnt1_.clear();

    const float *cur_kps0_gpu = currentKeypointsGpuBatch(0);
    const float *cur_kps1_gpu = currentKeypointsGpuBatch(1);
    if (!cur_kps0_gpu || !cur_kps1_gpu)
    {
        seedLeftFromCurrent(cur_dl_kps0_);
        return;
    }

    const int n_prev_l = static_cast<int>(prev_kf_left_kps_.size());

    // Batch element 0: temporal-left  (prev-left  -> cur-left)  — tracking.
    // Batch element 1: stereo         (cur-left   -> cur-right) — per-frame depth anchor.
    // Both run in a single matchBatched() launch on the b>=2 LightGlue engine.
    uosm::perception::DescriptorMatcher::MatchInputs in_l{
        d_prev_kf_kps_, n_prev_l, d_prev_kf_desc_,
        cur_kps0_gpu, cur_dl_n0_, cur_dl_d_desc0_,
        row_, col_};
    uosm::perception::DescriptorMatcher::MatchInputs in_s{
        cur_kps0_gpu, cur_dl_n0_, cur_dl_d_desc0_,
        cur_kps1_gpu, cur_dl_n1_, cur_dl_d_desc1_,
        row_, col_};

    std::vector<uosm::perception::DescriptorMatch> matches_l, matches_s;
    {
        auto results = descriptor_matcher_->matchBatched({in_l, in_s});
        if (results.size() == 2)
        {
            matches_l = std::move(results[0]);
            matches_s = std::move(results[1]);
        }
    }

    // Geometric outlier rejection.
    const int rej_l = filterMatchesE(matches_l, prev_kf_left_kps_, cur_dl_kps0_, m_cameras_[0]);
    current_metrics_.ransac_rejected_temporal += rej_l;

    // Temporal-left → cur_pts0_, ids0_, dl_to_pts0_idx_, dl_new_left_ids_.
    std::vector<int> assigned_l(cur_dl_n0_, -1), tcnt_l(cur_dl_n0_, 1);
    std::unordered_set<int> consumed_l;
    applyTemporalMatches(matches_l, prev_kf_left_ids_, prev_kf_left_track_cnt_,
                         cur_dl_n0_, assigned_l, tcnt_l, consumed_l);

    accumulateLeftOutputs(cur_dl_kps0_, assigned_l, tcnt_l);
    current_metrics_.forward_total = n_prev_l;
    current_metrics_.forward_success = static_cast<int>(consumed_l.size());

    // Stereo-LR anchors → cur_pts1_, ids1_ (inherited left IDs).
    pending_stereo_matches_ = std::move(matches_s);
    stereo_matches_pending_ = true;
    buildStereoAnchors();

    current_metrics_.matcher_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0)
            .count() /
        1000.0;
}

uint8_t FeatureTracker::acceptStereoOrphan(
    const cv::Point2f &l_und, const cv::Point2f &r_und,
    Eigen::Vector3d &P_cam0_out) const
{
    // Stereo acceptance gate
    constexpr float kDispMinPx = 1.0f;   // disparity floor: reject (near-)zero-disparity degenerate matches
    constexpr double kDepthMinM = 0.1;   // min plausible triangulated stereo depth [m]
    constexpr double kDepthMaxM = 100.0; // max plausible triangulated stereo depth [m]
    constexpr double kReprojMaxPx = 1.5; // max per-view stereo reprojection residual [px]

    const double sigma = 1.0 / static_cast<double>(focal_length_);
    const Eigen::Vector3d p_cam0 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d p_cam1 = t_cam0_cam1_;
    const Eigen::Vector3d e_cam0 = Eigen::Vector3d(l_und.x, l_und.y, 1.0).normalized();
    const Eigen::Vector3d e_cam1 = (R_cam0_cam1_ * Eigen::Vector3d(r_und.x, r_und.y, 1.0)).normalized();
    P_cam0_out = e_cam0; // Depth re-triangulated in FeatureManager

    // Accumulate every failing gate so the debug view can show rejections.
    uint8_t mask = StereoRejectReason::Ok;

    // Disparity floor: reject (near-)zero-disparity degenerate/false matches.
    const double disparity_px = std::abs(static_cast<double>(l_und.x - r_und.x)) * focal_length_;
    if (disparity_px < static_cast<double>(kDispMinPx))
        mask |= StereoRejectReason::Depth;

    bool is_valid = false;
    bool is_parallel = false;
    Eigen::Vector4d hp = triangulation::triangulateFast(
        p_cam0, e_cam0, p_cam1, e_cam1, sigma, is_valid, is_parallel);

    if (is_valid && !is_parallel)
    {
        const Eigen::Vector3d P_cam0 = hp.head<3>() / hp[3];
        const Eigen::Vector3d P_cam1 = R_cam0_cam1_.transpose() * (P_cam0 - t_cam0_cam1_);

        if (P_cam0.z() <= 0.0 || P_cam1.z() <= 0.0 ||
            P_cam0.z() < kDepthMinM || P_cam0.z() > kDepthMaxM)
        {
            mask |= StereoRejectReason::Depth;
        }
        else
        {
            // Reprojection residual: reproject the triangulated point back into both views.
            const Eigen::Vector2d pred_l(P_cam0.x() / P_cam0.z(), P_cam0.y() / P_cam0.z());
            const Eigen::Vector2d pred_r(P_cam1.x() / P_cam1.z(), P_cam1.y() / P_cam1.z());
            const double res_l = (pred_l - Eigen::Vector2d(l_und.x, l_und.y)).norm() * focal_length_;
            const double res_r = (pred_r - Eigen::Vector2d(r_und.x, r_und.y)).norm() * focal_length_;
            if (std::max(res_l, res_r) > kReprojMaxPx)
                mask |= StereoRejectReason::Reproj;
            else
                P_cam0_out = P_cam0; // only trust the point when no gate failed
        }
    }
    else if (!is_parallel)
    {
        // Non-parallel but triangulation is inconsistent (reprojects poorly) -> a genuine
        // bad match, not a far point.
        mask |= StereoRejectReason::Reproj;
    }

    return mask; // 0 == Ok
}

// Grow the persistent kept_rows scratch buffer if needed. Returns false on
// cudaMalloc failure; on success d_idx_persistent_ holds at least need_ints
// ints. Allocated once at max_tracked_keypoints_ and effectively never grown
// thereafter (n_kept is always <= max_kp).
bool FeatureTracker::ensureIdxPersistent(size_t need_ints)
{
    if (d_idx_persistent_ != nullptr && d_idx_persistent_capacity_ints_ >= need_ints)
        return true;
    if (d_idx_persistent_)
    {
        cudaFree(d_idx_persistent_);
        d_idx_persistent_ = nullptr;
        d_idx_persistent_capacity_ints_ = 0;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&d_idx_persistent_),
                   need_ints * sizeof(int)) != cudaSuccess)
    {
        d_idx_persistent_ = nullptr;
        return false;
    }
    d_idx_persistent_capacity_ints_ = need_ints;
    return true;
}

void FeatureTracker::buildStereoAnchors()
{
    cur_pts1_.clear();
    ids1_.clear();
    track_cnt1_.clear();
    stereo_match_viz_.clear();

    current_metrics_.stereo_candidates = 0;
    current_metrics_.stereo_success = 0;
    current_metrics_.stereo_gate_depth_rejects = 0;
    current_metrics_.stereo_gate_reproj_rejects = 0;

    if (!stereo_matches_pending_)
        return; // first frame / no prev cache — nothing to anchor
    auto matches = std::move(pending_stereo_matches_);
    stereo_matches_pending_ = false;
    current_metrics_.stereo_candidates = static_cast<int>(matches.size());
    if (matches.empty() || cur_dl_kps0_.empty() || cur_dl_kps1_.empty())
        return;

    auto t0 = std::chrono::high_resolution_clock::now();
    struct Cand
    {
        int idx0;   // DL left keypoint index
        int idx1;   // DL right keypoint index
        int best_l; // index into cur_pts0_/ids0_ of the tracked left feature
    };
    std::vector<Cand> cands;
    cands.reserve(matches.size());
    Point2fVec l_px, r_px;
    l_px.reserve(matches.size());
    r_px.reserve(matches.size());
    const int n0 = static_cast<int>(cur_dl_kps0_.size());
    const int n1 = static_cast<int>(cur_dl_kps1_.size());
    for (const auto &m : matches)
    {
        if (m.idx0 < 0 || m.idx1 < 0 || m.idx0 >= n0 || m.idx1 >= n1)
            continue;
        const int best_l = (m.idx0 < static_cast<int>(dl_to_pts0_idx_.size())) ? dl_to_pts0_idx_[m.idx0] : -1;
        if (best_l < 0 || best_l >= static_cast<int>(ids0_.size()))
            continue;
        cands.push_back({m.idx0, m.idx1, best_l});
        l_px.push_back(cur_dl_kps0_[m.idx0]);
        r_px.push_back(cur_dl_kps1_[m.idx1]);
    }
    if (cands.empty())
        return;

    const Point2fVec l_un = undistortPoints(l_px, m_cameras_[0]);
    const Point2fVec r_un = undistortPoints(r_px, m_cameras_[1]);
    if (l_un.size() != cands.size() || r_un.size() != cands.size())
        return;

    int depth_rejects = 0, reproj_rejects = 0, accepted = 0;
    for (size_t k = 0; k < cands.size(); ++k)
    {
        const Cand &c = cands[k];
        Eigen::Vector3d P_cam0;
        const uint8_t mask = acceptStereoOrphan(l_un[k], r_un[k], P_cam0);
        if (mask != StereoRejectReason::Ok)
        {
            if (mask & StereoRejectReason::Depth)
                ++depth_rejects;
            if (mask & StereoRejectReason::Reproj)
                ++reproj_rejects;
            stereo_match_viz_.push_back({c.idx0, c.idx1, static_cast<int>(mask)});
            continue;
        }

        cur_pts1_.push_back(cur_dl_kps1_[c.idx1]);
        ids1_.push_back(ids0_[c.best_l]); // inherit left feature ID
        track_cnt1_.push_back(c.best_l < static_cast<int>(track_cnt_.size())
                                  ? track_cnt_[c.best_l]
                                  : 1);
        stereo_match_viz_.push_back({c.idx0, c.idx1, 0});
        ++accepted;
    }

    const double dt_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::high_resolution_clock::now() - t0)
                             .count() /
                         1000.0;
    current_metrics_.matcher_time_ms += dt_ms;
    current_metrics_.stereo_success = accepted;
    current_metrics_.stereo_gate_depth_rejects = depth_rejects;
    current_metrics_.stereo_gate_reproj_rejects = reproj_rejects;
}

void FeatureTracker::updateKeyframeCache()
{
    if (!use_descriptor_matcher_)
        return;
    if (!cur_dl_valid_ || cur_dl_kps0_.empty() || cur_dl_d_desc0_ == nullptr || cur_dl_n0_ <= 0)
    {
        prev_kf_valid_ = false;
        return;
    }

    // Cache only DL keypoints that map to a tracked left-cam feature.
    Point2fVec kept_kps;
    std::vector<int> kept_ids;
    std::vector<int> kept_track_cnt;
    std::vector<int> kept_rows;
    kept_kps.reserve(cur_dl_kps0_.size());
    kept_ids.reserve(cur_dl_kps0_.size());
    kept_track_cnt.reserve(cur_dl_kps0_.size());
    kept_rows.reserve(cur_dl_kps0_.size());

    std::unordered_set<int> used_ids;
    for (size_t j = 0; j < cur_dl_kps0_.size(); ++j)
    {
        const int best_i = (j < dl_to_pts0_idx_.size()) ? dl_to_pts0_idx_[j] : -1;
        if (best_i < 0)
            continue;
        const int fid = ids0_[best_i];
        if (fid < 0 || used_ids.count(fid))
            continue;
        used_ids.insert(fid);
        kept_kps.push_back(cur_dl_kps0_[j]);
        kept_ids.push_back(fid);
        kept_track_cnt.push_back(best_i < static_cast<int>(track_cnt_.size())
                                     ? track_cnt_[best_i]
                                     : 1);
        kept_rows.push_back(static_cast<int>(j));
    }

    if (kept_kps.empty())
    {
        prev_kf_valid_ = false;
        return;
    }

    const int D = cur_dl_desc_dim_;
    const int n_kept = static_cast<int>(kept_rows.size());

    // Device-side cache of kept descriptors AND keypoints. Single gather kernel.
    // Outputs feed straight into the next frame's matcher without a host round-trip.
    if (D > 0)
    {
        const int max_kp = max_tracked_keypoints_;
        const size_t full_desc_bytes = static_cast<size_t>(max_kp) * D * sizeof(float);
        const size_t full_kps_bytes = static_cast<size_t>(max_kp) * kptInputDim() * sizeof(float);

        auto growDevBuf = [](float *&p, size_t &cap, size_t need) -> bool
        {
            if (need <= cap && p != nullptr)
                return true;
            if (p)
                cudaFree(p);
            if (cudaMalloc(reinterpret_cast<void **>(&p), need) != cudaSuccess)
            {
                p = nullptr;
                cap = 0;
                return false;
            }
            cap = need;
            return true;
        };

        if (!growDevBuf(d_prev_kf_desc_, d_prev_kf_desc_capacity_bytes_, full_desc_bytes) ||
            !growDevBuf(d_prev_kf_kps_, d_prev_kf_kps_capacity_bytes_, full_kps_bytes))
        {
            prev_kf_valid_ = false;
            RCLCPP_WARN(rclcpp::get_logger("feature_tracker"),
                        "updateKeyframeCache: cudaMalloc failed; matcher cache disabled");
            return;
        }

        // Zero entire buffers
        cudaMemsetAsync(d_prev_kf_desc_, 0, full_desc_bytes, /*stream=*/0);
        cudaMemsetAsync(d_prev_kf_kps_, 0, full_kps_bytes, /*stream=*/0);

        const float *kps_src = currentKeypointsGpuBatch(0);
        const float *desc_src = reinterpret_cast<const float *>(cur_dl_d_desc0_);
        if (kps_src && desc_src &&
            ensureIdxPersistent(static_cast<size_t>(n_kept)))
        {
            // H->D copy and gather both queue on stream 0. No sync here:
            cudaMemcpyAsync(d_idx_persistent_, kept_rows.data(),
                            static_cast<size_t>(n_kept) * sizeof(int),
                            cudaMemcpyHostToDevice, /*stream=*/0);
            launchGatherKeypointsAndDescriptors(
                kps_src, desc_src, d_idx_persistent_, n_kept, D, kptInputDim(),
                d_prev_kf_kps_, d_prev_kf_desc_, /*stream=*/0);
        }
    }

    // Host mirror of prev_kf_left_desc_ is only needed for VPR / loop closure consumers.
    // Skip it on the per-frame path; rebuild lazily if a downstream caller asks for it.
    prev_kf_left_desc_ = cv::Mat();

    for (auto &fid : kept_ids)
    {
        auto it = id_alias_.find(fid);
        if (it != id_alias_.end())
            fid = it->second;
    }
    prev_kf_left_kps_ = std::move(kept_kps);
    prev_kf_left_ids_ = std::move(kept_ids);
    prev_kf_left_track_cnt_ = std::move(kept_track_cnt);

    id_alias_.clear();
    prev_kf_valid_ = true;

    // Gather uses the default stream; ensure it completes before the next frame
    // overwrites d_sift_* source buffers or the matcher reads the prev cache.
    cudaStreamSynchronize(nullptr);
}

const float *FeatureTracker::currentKeypointsGpuBatch(int batch_idx) const
{
    if (feature_extractor_)
        return feature_extractor_->getKeypointsGpuBufferBatch(batch_idx);
    if (d_dkd_kpts_px_)
        return d_dkd_kpts_px_ +
               static_cast<size_t>(batch_idx) * max_tracked_keypoints_ * 2;
    if (d_sift_kps_)
        return d_sift_kps_ + static_cast<size_t>(batch_idx) * max_tracked_keypoints_ * kptInputDim();
    return nullptr;
}

bool FeatureTracker::initDenseAlikedComponents(int batch, bool need_dhead)
{
    if (batch <= 0)
        return false;

    if (dense_extractor_ && dense_extractor_->is_initialized() && dkd_buffers_ready_)
    {
        if (need_dhead && !(desc_head_ && desc_head_->is_initialized()))
        {
            // Build dhead now without re-allocating the dense extractor.
        }
        else
        {
            return true;
        }
    }

    const std::string role = use_stereo_mode_ ? "stereo_extractor" : "mono_extractor";

    if (!dense_extractor_)
    {
        std::string dense_engine = resolveEnginePath(role,
                                                     {"aliked_n16", "dense"},
                                                     /*max_kp=*/-1);
        if (dense_engine.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                         "ALIKED dense: aliked_n16[rot]_dense engine not found under '%s/%s'",
                         weights_folder_.c_str(), role.c_str());
            return false;
        }
        uosm::perception::DenseExtractor::Params dparams;
        dparams.engine_path = dense_engine;
        dparams.batch_size = batch;
        dparams.input_h = row_;
        dparams.input_w = col_;
        dparams.input_channels = 3;
        dparams.feature_dim = 128;
        dparams.profile_inference = profile_trt_inference_;
        dense_extractor_ = std::make_unique<uosm::perception::DenseExtractor>(dparams);
        if (!dense_extractor_ || !dense_extractor_->is_initialized())
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "DenseExtractor init failed");
            dense_extractor_.reset();
            return false;
        }
    }

    if (need_dhead && !desc_head_)
    {
        std::string dhead_engine = resolveEnginePath(role,
                                                     {"aliked_n16", "dhead"},
                                                     max_tracked_keypoints_);
        if (dhead_engine.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                         "ALIKED dense: aliked_n16[rot]_dhead engine not found under '%s/%s'",
                         weights_folder_.c_str(), role.c_str());
            return false;
        }
        uosm::perception::DescriptorHead::Params hparams;
        hparams.engine_path = dhead_engine;
        hparams.batch = batch;
        hparams.max_kp = max_tracked_keypoints_;
        hparams.model_h = dense_extractor_->getInputH();
        hparams.model_w = dense_extractor_->getInputW();
        hparams.feature_dim = dense_extractor_->getFeatureDim();
        desc_head_ = std::make_unique<uosm::perception::DescriptorHead>(hparams);
        if (!desc_head_ || !desc_head_->is_initialized())
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "DescriptorHead init failed");
            desc_head_.reset();
            return false;
        }
    }

    // Allocate GPU resources only on first call (when dkd_buffers_ready_ is false).
    if (!dkd_buffers_ready_)
    {
        cudaError_t err = cudaStreamCreate(&dl_chain_stream_);
        if (err != cudaSuccess)
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                         "ALIKED dense: cudaStreamCreate failed: %s", cudaGetErrorString(err));
            return false;
        }

        const int H = dense_extractor_->getInputH();
        const int W = dense_extractor_->getInputW();
        const int K = max_tracked_keypoints_;

        d_dkd_workspace_bytes_ = alikedDkdPostprocWorkspaceBytes(batch, H, W, K);
        if (cudaMalloc(&d_dkd_workspace_, d_dkd_workspace_bytes_) != cudaSuccess)
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                         "ALIKED dense: cudaMalloc DKD workspace (%zu bytes) failed",
                         d_dkd_workspace_bytes_);
            return false;
        }
        cudaMalloc(reinterpret_cast<void **>(&d_dkd_kpts_px_), batch * K * 2 * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&d_dkd_kpts_n_), batch * K * 2 * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&d_dkd_scores_), batch * K * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&d_dkd_num_kpts_), batch * sizeof(int));
        if (!d_dkd_kpts_px_ || !d_dkd_kpts_n_ || !d_dkd_scores_ || !d_dkd_num_kpts_)
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                         "ALIKED dense: cudaMalloc output buffers failed");
            return false;
        }

        // Pinned host mirror for kpts_px + scores + num_kpts (descriptors stay
        // on GPU and are consumed via cur_dl_d_desc{0,1}_).
        const size_t pinned_bytes = batch * K * 2 * sizeof(float) // kpts_px
                                    + batch * K * sizeof(float)   // scores
                                    + batch * sizeof(int);        // num_kpts
        dkd_pinned_host_ = cv::cuda::HostMem(1, static_cast<int>(pinned_bytes), CV_8UC1,
                                             cv::cuda::HostMem::PAGE_LOCKED);
        dkd_buffers_ready_ = true;
    }
    return true;
}

void FeatureTracker::featureExtractAliked()
{
    auto extract_start = std::chrono::high_resolution_clock::now();

    if (!dense_extractor_ || !dense_extractor_->is_initialized() || !dkd_buffers_ready_)
    {
        RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"),
                     "ALIKED dense: pipeline not initialized");
        n_pts_.clear();
        return;
    }

    // Reset per-frame DL cache (populated below).
    cur_dl_valid_ = false;
    cur_dl_stereo_ = false;
    cur_dl_kps0_.clear();
    cur_dl_kps1_.clear();
    cur_dl_desc0_ = cv::Mat();
    cur_dl_d_desc0_ = nullptr;
    cur_dl_d_desc1_ = nullptr;
    cur_dl_n0_ = 0;
    cur_dl_n1_ = 0;
    cur_dl_host_desc0_valid_ = false;

    const int B = dense_extractor_->getBatch();
    const int K = max_tracked_keypoints_;
    const int H = dense_extractor_->getInputH();
    const int W = dense_extractor_->getInputW();

    const bool run_stereo = use_stereo_mode_ && (B == 2) && !cur_img1_.empty();

    // 1. Dense extractor (image -> score_map + feature_map) on dl_chain_stream_.
    if (run_stereo)
    {
        if (cur_gpu_img0_.empty())
            cur_gpu_img0_.upload(cur_img0_);
        if (cur_gpu_img1_.empty())
            cur_gpu_img1_.upload(cur_img1_);
        if (!dense_extractor_->runInferenceStereoFromGPU(cur_gpu_img0_, cur_gpu_img1_, dl_chain_stream_))
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "ALIKED dense: stereo inference failed");
            n_pts_.clear();
            return;
        }
    }
    else
    {
        if (cur_gpu_img0_.empty())
            cur_gpu_img0_.upload(cur_img0_);
        if (!dense_extractor_->runInferenceFromGPU(cur_gpu_img0_, dl_chain_stream_))
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "ALIKED dense: mono inference failed");
            n_pts_.clear();
            return;
        }
    }

    // 2. custom CUDA DKD postproc on dl_chain_stream_.
    launchAlikedDkdPostproc(
        dense_extractor_->getScoreMapGpu(),
        B, H, W, K,
        /*nms_radius*/ 2,
        /*scores_th*/ 0.0f,
        /*subpixel_T*/ 0.1f,
        dense_extractor_->getScale(),
        dense_extractor_->getXOffset(),
        dense_extractor_->getYOffset(),
        d_dkd_kpts_px_, d_dkd_kpts_n_, d_dkd_scores_, d_dkd_num_kpts_,
        d_dkd_workspace_, d_dkd_workspace_bytes_,
        dl_chain_stream_);

    // 3. desc-head-only engine on dl_chain_stream_ (no internal sync).
    // Gated on use_descriptor_matcher_: LK pathway never reads descriptors.
    const bool need_descs = use_descriptor_matcher_ && desc_head_ && desc_head_->is_initialized();
    if (need_descs)
    {
        if (!desc_head_->runInference(dense_extractor_->getFeatureMapGpu(), d_dkd_kpts_n_, dl_chain_stream_))
        {
            RCLCPP_ERROR(rclcpp::get_logger("feature_tracker"), "ALIKED dense: dhead inference failed");
            n_pts_.clear();
            return;
        }
    }

    unsigned char *host_base = dkd_pinned_host_.createMatHeader().data;
    const size_t kpts_px_bytes = B * K * 2 * sizeof(float);
    const size_t scores_bytes = B * K * sizeof(float);
    const size_t num_kpts_bytes = B * sizeof(int);
    float *h_kpts_px = reinterpret_cast<float *>(host_base);
    int *h_num_kpts = reinterpret_cast<int *>(host_base + kpts_px_bytes + scores_bytes);
    cudaMemcpyAsync(h_kpts_px, d_dkd_kpts_px_, kpts_px_bytes, cudaMemcpyDeviceToHost, dl_chain_stream_);
    cudaMemcpyAsync(h_num_kpts, d_dkd_num_kpts_, num_kpts_bytes, cudaMemcpyDeviceToHost, dl_chain_stream_);
    cudaStreamSynchronize(dl_chain_stream_);

    // Build host vectors + cache GPU descriptor pointers.
    const int n0 = std::max(0, std::min(K, h_num_kpts[0]));
    const int n1 = (B >= 2) ? std::max(0, std::min(K, h_num_kpts[1])) : 0;

    cur_dl_kps0_.clear();
    cur_dl_kps0_.reserve(n0);
    for (int i = 0; i < n0; ++i)
    {
        cur_dl_kps0_.emplace_back(h_kpts_px[i * 2 + 0], h_kpts_px[i * 2 + 1]);
    }
    if (B >= 2)
    {
        cur_dl_kps1_.clear();
        cur_dl_kps1_.reserve(n1);
        for (int i = 0; i < n1; ++i)
        {
            cur_dl_kps1_.emplace_back(h_kpts_px[(K + i) * 2 + 0], h_kpts_px[(K + i) * 2 + 1]);
        }
    }
    cur_dl_n0_ = n0;
    cur_dl_n1_ = n1;
    cur_dl_desc_dim_ = need_descs ? desc_head_->getDescriptorDim() : 0;
    cur_dl_stereo_ = run_stereo;
    cur_dl_valid_ = true;

    // Descriptor pointers per batch (descriptors stay on GPU). Only populated
    // when dhead actually ran; LK path leaves them null.
    if (need_descs)
    {
        const size_t per_batch_bytes = static_cast<size_t>(K) * cur_dl_desc_dim_ * sizeof(float);
        char *desc_base = static_cast<char *>(desc_head_->getDescriptorsGpu());
        if (n0 > 0)
            cur_dl_d_desc0_ = desc_base;
        if (B >= 2 && n1 > 0)
            cur_dl_d_desc1_ = desc_base + per_batch_bytes;
    }

    // DL-temporal pipeline consumes only cur_dl_* caches; it clears
    // cur_pts0_/cur_pts1_ and never reads n_pts_. Skip setMask + top-up loop.
    n_pts_.clear();
    if (dlDescriptorTemporalActive())
    {
        recordExtractionTime(extract_start);
        return;
    }

    setMask();
    const int n_needed = max_tracked_keypoints_ - static_cast<int>(cur_pts0_.size());
    if (n_needed <= 0)
    {
        recordExtractionTime(extract_start);
        return;
    }

    // Filter keypoints by min distance from existing tracked points
    for (size_t i = 0; i < cur_dl_kps0_.size() && static_cast<int>(n_pts_.size()) < n_needed; ++i)
    {
        const cv::Point2f &candidate = cur_dl_kps0_[i];
        if (!inBorder(candidate))
            continue;
        int px = cvRound(candidate.x);
        int py = cvRound(candidate.y);
        if (mask_.at<uchar>(py, px) == 0)
            continue;

        bool too_close = false;
        for (const auto &existing : cur_pts0_)
        {
            if (distance_sq(candidate, existing) < min_feature_distance_ * min_feature_distance_)
            {
                too_close = true;
                break;
            }
        }
        if (too_close)
            continue;
        n_pts_.push_back(candidate);
    }

    recordExtractionTime(extract_start);
}
