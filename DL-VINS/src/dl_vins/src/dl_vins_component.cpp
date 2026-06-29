#include "../include/dl_vins_component.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <cstring>
#include <filesystem>

namespace uosm
{
    namespace perception
    {
        DlVinsComponent::DlVinsComponent(const rclcpp::NodeOptions &options) : Node("dl_vins_component", options)
        {
            RCLCPP_INFO(get_logger(), "**********************************");
            RCLCPP_INFO(get_logger(), " DLVINS Component                 ");
            RCLCPP_INFO(get_logger(), "**********************************");
            RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
            RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
            RCLCPP_INFO(get_logger(), "**********************************");

            // Estimator parameters
            estimator_config_.gravity_magnitude = declare_parameter<double>("gravity", 9.81f);
            estimator_config_.acc_n = declare_parameter<double>("acc_noise", 0.1f);
            estimator_config_.gyr_n = declare_parameter<double>("gyr_noise", 0.01f);
            estimator_config_.acc_w = declare_parameter<double>("acc_bias_noise", 0.001f);
            estimator_config_.gyr_w = declare_parameter<double>("gyr_bias_noise", 0.0001f);
            estimator_config_.max_reprojection_error = declare_parameter<double>("max_reprojection_error", 3.0f);
            estimator_config_.td = declare_parameter<double>("initial_td", 0.0f);
            estimator_config_.estimate_td = declare_parameter<bool>("estimate_td", false);
            estimator_config_.max_iterations = declare_parameter<int>("max_iterations", 8U);
            estimator_config_.max_solver_time = declare_parameter<double>("max_solver_time", 0.04f);
            estimator_config_.estimate_extrinsic = declare_parameter<bool>("estimate_extrinsic", false);
            estimator_config_.failure_recovery = declare_parameter<bool>("failure_recovery", false);
            estimator_config_.num_threads = declare_parameter<int>("num_threads", 0U);
            estimator_config_.use_gpu = (declare_parameter<int>("ceres_device", 0) == 1);

            // Init controls
            estimator_config_.init_max_g_error = declare_parameter<double>("init_max_g_error", 0.5f);
            estimator_config_.init_max_abias_norm = declare_parameter<double>("init_max_abias_norm", 0.0f);
            estimator_config_.init_min_scale = declare_parameter<double>("init_min_scale", 0.1f);
            estimator_config_.init_max_scale = declare_parameter<double>("init_max_scale", 10.0f);
            estimator_config_.init_min_parallax_ratio =
                declare_parameter<double>("init_min_parallax_ratio", 30.0 / REF_FOCAL_PX);
            odom_frame_id_ = declare_parameter<std::string>("odom_frame_id", "odom");
            norm_focal_length_ = declare_parameter<double>("norm_focal_length", 0.0f);

            // Feature tracker parameters
            tracker_config_.allow_flowback = declare_parameter<bool>("use_flowback", true);
            tracker_config_.enable_visualization = declare_parameter<bool>("enable_visualization", false);
            tracker_config_.calib_file_cam0 = declare_parameter<std::string>("cam0_calib", "");
            tracker_config_.calib_file_cam1 = declare_parameter<std::string>("cam1_calib", "");
            tracker_config_.col = declare_parameter<int>("image_width", 752U);
            tracker_config_.row = declare_parameter<int>("image_height", 480U);
            image_encoding_ = declare_parameter<std::string>("image_encoding", "mono8");
            tracker_config_.min_feature_distance = declare_parameter<int>("min_feature_distance", 10U);
            tracker_config_.max_tracked_keypoints = declare_parameter<int>("max_tracked_keypoints", 256U);
            tracker_config_.optflow_max_iterations = declare_parameter<int>("optflow_max_iterations", 30U);
            tracker_config_.optflow_pyramid_levels = declare_parameter<int>("optflow_pyramid_levels", 3U);
            tracker_config_.optflow_window_dim = declare_parameter<int>("optflow_window_dim", 21U);
            tracker_config_.optflow_pyramid_scale = static_cast<float>(declare_parameter<double>("optflow_pyramid_scale", 0.5f));
            tracker_config_.optflow_epsilon = static_cast<float>(declare_parameter<double>("optflow_epsilon", 0.01f));
            tracker_config_.feature_extraction_method =
                static_cast<FeatureTracker::FeatureExtractionMethod>(declare_parameter<int>("feature_extraction_method", 0));
            tracker_config_.optflow_device =
                static_cast<FeatureTracker::OpticalFlowBackend>(declare_parameter<int>("optflow_device", 0));
            tracker_config_.weights_folder = declare_parameter<std::string>("weights_folder", "");
            tracker_config_.use_descriptor_matcher = declare_parameter<bool>("use_descriptor_matcher", false);
            tracker_config_.matcher_type = declare_parameter<std::string>("matcher_type", "lightglue");
            tracker_config_.matcher_score_threshold =
                static_cast<float>(declare_parameter<double>("matcher_score_threshold", 0.0f));
            tracker_config_.profile_trt_inference = declare_parameter<bool>("profile_trt_inference", false);
            tracker_config_.stereo_window_dim = declare_parameter<int>("stereo_window_dim", 21U);
            tracker_config_.stereo_pyramid_levels = declare_parameter<int>("stereo_pyramid_levels", 3U);
            tracker_config_.raco_cov_alpha = static_cast<float>(declare_parameter<double>("raco_cov_alpha", 1.0f));
            tracker_config_.raco_cov_floor_px = static_cast<float>(declare_parameter<double>("raco_cov_floor_px", 0.0f));
            tracker_config_.fisheye = declare_parameter<bool>("fisheye", false);
            tracker_config_.fisheye_mask = declare_parameter<std::string>("fisheye_mask", "");

            use_stereo_ = !tracker_config_.calib_file_cam1.empty();
            estimator_config_.use_stereo = use_stereo_;

            enable_csv_logging_ = declare_parameter<bool>("enable_csv_logging", false);
            log_folder_ = declare_parameter<std::string>("log_folder", "tmp/dl_vins_logs");

            // Extrinsic calibration (cam1 only declared in stereo mode)
            const std::vector<double> default_identity = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            body_T_cam0_ = declare_parameter<std::vector<double>>("body_T_cam0", default_identity);
            if (use_stereo_)
            {
                body_T_cam1_ = declare_parameter<std::vector<double>>("body_T_cam1", default_identity);
            }

            RCLCPP_INFO(get_logger(), "Estimator Parameters:");
            RCLCPP_INFO(get_logger(), " * window_size: %d (compile-time)", WINDOW_SIZE);
            RCLCPP_INFO(get_logger(), " * gravity: %.5f", estimator_config_.gravity_magnitude);
            RCLCPP_INFO(get_logger(), " * device: %s", estimator_config_.use_gpu ? "GPU" : "CPU");
            RCLCPP_INFO(get_logger(), " * mode: %s", use_stereo_ ? "stereo" : "mono");
            RCLCPP_INFO(get_logger(), " * initial_td: %.4f s (estimate_td=%s)",
                        estimator_config_.td, estimator_config_.estimate_td ? "true" : "false");
            if (use_stereo_)
            {
                RCLCPP_INFO(get_logger(), " * init: gyro-bias-only (stereo), max_g_error=%.3f",
                            estimator_config_.init_max_g_error);
            }
            else
            {
                RCLCPP_INFO(get_logger(), " * init gates: max_g_error=%.3f, max_abias=%.3f, scale=[%.3f, %.3f]",
                            estimator_config_.init_max_g_error,
                            estimator_config_.init_max_abias_norm,
                            estimator_config_.init_min_scale,
                            estimator_config_.init_max_scale);
                RCLCPP_INFO(get_logger(), " * init_min_parallax_ratio: %.4f (%.1f px @ ref focal %.0f)",
                            estimator_config_.init_min_parallax_ratio,
                            estimator_config_.init_min_parallax_ratio * REF_FOCAL_PX, REF_FOCAL_PX);
            }
            RCLCPP_INFO(get_logger(), " * estimate_extrinsic: %s",
                        estimator_config_.estimate_extrinsic ? "true (online, gated)" : "false (fixed)");

            const char *extraction_methods[] = {
                "GFTT_CPU", "GFTT_CUDA", "ALIKED", "SuperPoint", "RACO", "XFeat", "SIFT_CPU"};
            const char *optflow_devices[] = {"CPU", "CUDA"};
            const int em_idx = std::clamp(static_cast<int>(tracker_config_.feature_extraction_method), 0,
                                          static_cast<int>(std::size(extraction_methods)) - 1);
            const int of_idx = std::clamp(static_cast<int>(tracker_config_.optflow_device), 0,
                                          static_cast<int>(std::size(optflow_devices)) - 1);
            RCLCPP_INFO(get_logger(), "Tracker Parameters:");
            RCLCPP_INFO(get_logger(), " * feature_extraction: %s", extraction_methods[em_idx]);
            if (!tracker_config_.use_descriptor_matcher)
                RCLCPP_INFO(get_logger(), " * optflow_device: %s", optflow_devices[of_idx]);
            RCLCPP_INFO(get_logger(), " * max_keypoints: %d", tracker_config_.max_tracked_keypoints);
            RCLCPP_INFO(get_logger(), " * use_descriptor_matcher: %s", tracker_config_.use_descriptor_matcher ? "true" : "false");

            std::chrono::milliseconds init_msec(10U);
            init_timer_ = create_wall_timer(init_msec, std::bind(&DlVinsComponent::init, this));
        }

        DlVinsComponent::~DlVinsComponent()
        {
            RCLCPP_WARN(get_logger(), "DlVinsComponent shutting down.");

            // Stop tracking thread first
            {
                tracking_thread_.request_stop();
                image_buf_cv_.notify_all();
            }

            if (estimator_)
            {
                estimator_->stopProcessThread();
            }
            uosm::utility::MetricsLogger::getInstance().shutdown();
        }

        void DlVinsComponent::init()
        {
            if (init_timer_)
            {
                init_timer_->cancel();
                init_timer_.reset();
            }

            // Resolve relative paths for calibration files
            std::string package_share_dir;
            try
            {
                package_share_dir = ament_index_cpp::get_package_share_directory("dl_vins");
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(get_logger(), "Could not get package share directory: %s", e.what());
            }

            auto resolvePath = [&package_share_dir](const std::string &path) -> std::string
            {
                if (path.empty())
                    return path;
                std::filesystem::path p(path);
                if (p.is_relative() && !package_share_dir.empty())
                {
                    return (std::filesystem::path(package_share_dir) / p).string();
                }
                return path;
            };

            tracker_config_.calib_file_cam0 = resolvePath(tracker_config_.calib_file_cam0);
            tracker_config_.calib_file_cam1 = resolvePath(tracker_config_.calib_file_cam1);
            tracker_config_.weights_folder = resolvePath(tracker_config_.weights_folder);
            tracker_config_.fisheye_mask = resolvePath(tracker_config_.fisheye_mask);

            RCLCPP_INFO(get_logger(), "Resolved paths:");
            RCLCPP_INFO(get_logger(), "  cam0_calib: %s", tracker_config_.calib_file_cam0.c_str());
            RCLCPP_INFO(get_logger(), "  fisheye: %s, fisheye_mask: %s",
                        tracker_config_.fisheye ? "true" : "false",
                        tracker_config_.fisheye_mask.c_str());
            if (use_stereo_)
            {
                RCLCPP_INFO(get_logger(), "  cam1_calib: %s", tracker_config_.calib_file_cam1.c_str());
            }
            RCLCPP_INFO(get_logger(), "  weights_folder: %s", tracker_config_.weights_folder.c_str());
            RCLCPP_INFO(get_logger(), "  use_descriptor_matcher: %s (type='%s')",
                        tracker_config_.use_descriptor_matcher ? "true" : "false",
                        tracker_config_.matcher_type.c_str());

            // Configure metrics logging
            uosm::utility::MetricsConfig metrics_config;
            metrics_config.enable_csv_logging = enable_csv_logging_;
            metrics_config.log_folder = log_folder_;
            uosm::utility::MetricsLogger::getInstance().configure(metrics_config);

            feature_tracker_ = std::make_unique<FeatureTracker>(tracker_config_);
            RCLCPP_INFO(get_logger(), "Feature tracker initialized");

            // Load extrinsic calibration
            if (body_T_cam0_.size() != 16 || (use_stereo_ && body_T_cam1_.size() != 16))
            {
                RCLCPP_ERROR(get_logger(), "Invalid extrinsic calibration size.");
                throw std::runtime_error("Invalid extrinsic calibration: body_T_cam0 (and body_T_cam1 in stereo) must each have 16 elements");
            }

            const Eigen::Matrix4d T_body_cam0 =
                Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(body_T_cam0_.data());
            const Eigen::Matrix3d ric0 = T_body_cam0.block<3, 3>(0, 0);
            const Eigen::Vector3d tic0 = T_body_cam0.block<3, 1>(0, 3);

            RCLCPP_INFO(get_logger(), "Extrinsics loaded:");
            RCLCPP_INFO(get_logger(), "  tic0: [%.4f, %.4f, %.4f]", tic0.x(), tic0.y(), tic0.z());

            Eigen::Matrix3d ric1 = Eigen::Matrix3d::Identity();
            Eigen::Vector3d tic1 = Eigen::Vector3d::Zero();

            if (use_stereo_)
            {
                const Eigen::Matrix4d T_body_cam1 =
                    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(body_T_cam1_.data());
                ric1 = T_body_cam1.block<3, 3>(0, 0);
                tic1 = T_body_cam1.block<3, 1>(0, 3);

                const double stereo_baseline = (tic1 - tic0).norm();
                RCLCPP_INFO(get_logger(), "  tic1: [%.4f, %.4f, %.4f]", tic1.x(), tic1.y(), tic1.z());
                RCLCPP_INFO(get_logger(), "  stereo_baseline: %.4f m", stereo_baseline);

                feature_tracker_->setStereoBaseline(static_cast<float>(stereo_baseline));
                const Eigen::Matrix3d R_cam0_cam1 = ric0.transpose() * ric1;
                const Eigen::Vector3d t_cam0_cam1 = ric0.transpose() * (tic1 - tic0);
                feature_tracker_->setStereoExtrinsics(R_cam0_cam1, t_cam0_cam1);
            }

            // Normalization focal length: YAML override > auto-detect from camera model
            const double auto_focal = static_cast<double>(feature_tracker_->getFocalLength());
            if (norm_focal_length_ > 0.0)
            {
                RCLCPP_INFO(get_logger(), "Using YAML norm_focal_length: %.2f px", norm_focal_length_);
            }
            else if (auto_focal > 0.0)
            {
                norm_focal_length_ = auto_focal;
                RCLCPP_INFO(get_logger(), "Auto-detected norm_focal_length: %.2f px", norm_focal_length_);
            }
            else
            {
                norm_focal_length_ = 460.0;
                RCLCPP_WARN(get_logger(), "Could not detect focal length, falling back to %.1f", norm_focal_length_);
            }
            estimator_config_.focal_length = norm_focal_length_;
            estimator_ = std::make_unique<Estimator>(estimator_config_);

            std::array<Eigen::Matrix3d, 2> ric_arr = {ric0, ric1};
            std::array<Eigen::Vector3d, 2> tic_arr = {tic0, tic1};
            estimator_->setExtrinsic(ric_arr, tic_arr);

            estimator_->startProcessThread();

            tracking_thread_ = std::jthread([this](std::stop_token st)
                                            { trackingLoop(st); });
            RCLCPP_INFO(get_logger(), "Tracking thread started");

            // IMU subscriber.
            const std::string imu_qos_rel =
                declare_parameter<std::string>("imu_qos_reliability", "reliable");
            rclcpp::QoS imu_qos = rclcpp::QoS(200).durability_volatile();
            if (imu_qos_rel == "reliable")
                imu_qos.reliable();
            else
                imu_qos.best_effort();
            RCLCPP_INFO(get_logger(), "IMU subscription QoS reliability: %s", imu_qos_rel.c_str());

            rclcpp::SubscriptionOptions imu_options;
            imu_options.qos_overriding_options =
                rclcpp::QosOverridingOptions::with_default_policies();
            imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
                "imu",
                imu_qos,
                std::bind(&DlVinsComponent::imuCallback, this, std::placeholders::_1),
                imu_options);

            if (use_stereo_)
            {
                // Stereo image subscribers with synchronization
                image0_sub_.subscribe(this, "image0", rmw_qos_profile_sensor_data);
                image1_sub_.subscribe(this, "image1", rmw_qos_profile_sensor_data);
                image_sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                    SyncPolicy(10), image0_sub_, image1_sub_);
                image_sync_->setMaxIntervalDuration(rclcpp::Duration(std::chrono::milliseconds(1)));
                image_sync_->registerCallback(&DlVinsComponent::imagesCallback, this);
            }
            else
            {
                image_mono_sub_ = create_subscription<sensor_msgs::msg::Image>(
                    "image0",
                    rclcpp::SensorDataQoS(),
                    std::bind(&DlVinsComponent::imageCallback, this, std::placeholders::_1));
            }

            // Publishers
            imu_propagate_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("dl_vins/imu_propagate_odom", 30);
            odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("dl_vins/odometry", 30);
            path_pub_ = create_publisher<nav_msgs::msg::Path>("dl_vins/path", 10);
            point_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("dl_vins/point_cloud", 10);

            if (tracker_config_.enable_visualization)
            {
                track_viz_pub_ = create_publisher<sensor_msgs::msg::Image>("image_track", 10);
            }

            descriptor_pub_ = create_publisher<dl_vins::msg::FrameDescriptors>("dl_vins/frame_descriptors", 10);
            keyframe_pose_pub_ = create_publisher<nav_msgs::msg::Odometry>("/keyframe_pose", 100);
            keyframe_point_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/keyframe_point", 100);
            extrinsic_pub_ = create_publisher<nav_msgs::msg::Odometry>("/extrinsic", 100);
            margin_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/margin_cloud", 100);

            RCLCPP_INFO(get_logger(), "DlVinsComponent fully initialized");
        }

        void DlVinsComponent::imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &imu_msg)
        {
            double t = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9;
            Eigen::Vector3d acc(imu_msg->linear_acceleration.x,
                                imu_msg->linear_acceleration.y,
                                imu_msg->linear_acceleration.z);
            Eigen::Vector3d gyr(imu_msg->angular_velocity.x,
                                imu_msg->angular_velocity.y,
                                imu_msg->angular_velocity.z);

            if (estimator_)
            {
                estimator_->inputIMU(t, acc, gyr);

                if (estimator_->isInitialized() &&
                    imu_propagate_odom_pub_->get_subscription_count() > 0)
                {
                    Eigen::Vector3d P, V;
                    Eigen::Quaterniond Q;
                    estimator_->getIMUPropagatedState(P, Q, V);
                    publishIMUOdometry(imu_msg->header, P, Q, V);
                }
            }
        }

        void DlVinsComponent::imagesCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg,
                                             const sensor_msgs::msg::Image::ConstSharedPtr &image1_msg)
        {
            if (!feature_tracker_)
                return;

            try
            {
                cv_bridge::CvImageConstPtr image0_cv, image1_cv;
                const std::string &encoding = image_encoding_;

                if (encoding.empty() || image0_msg->encoding == encoding)
                {
                    image0_cv = cv_bridge::toCvShare(image0_msg, encoding.empty() ? image0_msg->encoding : encoding);
                    image1_cv = cv_bridge::toCvShare(image1_msg, encoding.empty() ? image1_msg->encoding : encoding);
                }
                else
                {
                    image0_cv = cv_bridge::toCvCopy(image0_msg, encoding);
                    image1_cv = cv_bridge::toCvCopy(image1_msg, encoding);
                }

                if (image0_cv->image.empty() || image1_cv->image.empty())
                {
                    RCLCPP_WARN(get_logger(), "Received empty image(s)");
                    return;
                }

                double timestamp = image0_msg->header.stamp.sec + image0_msg->header.stamp.nanosec * 1e-9;

                StampedImagePair pair;
                pair.timestamp = timestamp;
                pair.img0 = image0_cv->image.clone();
                pair.img1 = image1_cv->image.clone();
                pair.header = image0_msg->header;

                {
                    std::lock_guard lock(image_buf_mutex_);
                    image_buf_.push(std::move(pair));
                }
                image_buf_cv_.notify_one();
            }
            catch (const cv_bridge::Exception &e)
            {
                RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(get_logger(), "Exception in imagesCallback: %s", e.what());
            }
        }

        void DlVinsComponent::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg)
        {
            if (!feature_tracker_)
                return;

            try
            {
                cv_bridge::CvImageConstPtr image0_cv;
                const std::string &encoding = image_encoding_;

                if (encoding.empty() || image0_msg->encoding == encoding)
                {
                    image0_cv = cv_bridge::toCvShare(image0_msg, encoding.empty() ? image0_msg->encoding : encoding);
                }
                else
                {
                    image0_cv = cv_bridge::toCvCopy(image0_msg, encoding);
                }

                if (image0_cv->image.empty())
                {
                    RCLCPP_WARN(get_logger(), "Received empty image");
                    return;
                }

                double timestamp = image0_msg->header.stamp.sec + image0_msg->header.stamp.nanosec * 1e-9;

                StampedImagePair pair;
                pair.timestamp = timestamp;
                pair.img0 = image0_cv->image.clone();
                pair.header = image0_msg->header;

                {
                    std::lock_guard lock(image_buf_mutex_);
                    image_buf_.push(std::move(pair));
                }
                image_buf_cv_.notify_one();
            }
            catch (const cv_bridge::Exception &e)
            {
                RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(get_logger(), "Exception in imageCallback: %s", e.what());
            }
        }

        void DlVinsComponent::trackingLoop(std::stop_token st)
        {
            while (!st.stop_requested())
            {
                StampedImagePair pair;
                {
                    std::unique_lock lock(image_buf_mutex_);
                    image_buf_cv_.wait(lock, st, [&]
                                       { return !image_buf_.empty(); });
                    if (st.stop_requested())
                        break;
                    pair = std::move(image_buf_.front());
                    image_buf_.pop();
                }

                cv::Mat img0_in = pair.img0;
                cv::Mat img1_in = use_stereo_ ? pair.img1 : cv::Mat();

                const int next_image_count = input_image_count_ + 1;
                const bool is_decimated_input = (next_image_count % 2 == 0);

                auto observations = feature_tracker_->trackImage(pair.timestamp, img0_in, img1_in);

                if (observations.empty())
                {
                    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("tracking_thread"),
                                         *get_clock(), 1000, "No features tracked");
                    continue;
                }

                input_image_count_ = next_image_count;

                if (estimator_)
                {
                    if (is_decimated_input)
                    {
                        estimator_->inputFeatures(pair.timestamp, observations);
                    }

                    if (estimator_->isInitialized())
                    {
                        const uint64_t backend_seq = estimator_->getBackendUpdateSeq();
                        if (backend_seq != last_published_backend_seq_)
                        {
                            last_published_backend_seq_ = backend_seq;
                            publishOdometry(pair.header);
                        }
                        // Keyframe artefacts (loop closure, etc.) stay keyframe-only.
                        const uint64_t kf_seq = estimator_->getKeyframeSeq();
                        if (kf_seq != last_published_keyframe_seq_)
                        {
                            last_published_keyframe_seq_ = kf_seq;
                            publishKeyframeArtifacts();
                        }
                        publishPath(pair.header);
                        publishPointCloud(pair.header);
                    }
                }

                if (descriptor_pub_ && descriptor_pub_->get_subscription_count() > 0)
                {
                    const auto &desc = feature_tracker_->getCurrentDescriptors();
                    const auto &desc_ids = feature_tracker_->getDescriptorIds();
                    if (!desc.empty() && !desc_ids.empty())
                    {
                        auto desc_msg = dl_vins::msg::FrameDescriptors();
                        desc_msg.header = pair.header;
                        desc_msg.descriptor_dim = static_cast<uint32_t>(desc.cols);
                        desc_msg.feature_ids.assign(desc_ids.begin(), desc_ids.end());
                        const size_t payload_bytes = desc.total() * desc.elemSize();
                        desc_msg.descriptors.resize(payload_bytes / sizeof(float));
                        if (desc.isContinuous())
                        {
                            std::memcpy(desc_msg.descriptors.data(), desc.data, payload_bytes);
                        }
                        else
                        {
                            cv::Mat desc_cont = desc.clone();
                            std::memcpy(desc_msg.descriptors.data(), desc_cont.data, payload_bytes);
                        }
                        descriptor_pub_->publish(desc_msg);
                    }
                }

                if (tracker_config_.enable_visualization && track_viz_pub_ &&
                    track_viz_pub_->get_subscription_count() > 0)
                {
                    auto viz_image = feature_tracker_->getTrackVisualization();
                    if (!viz_image.empty())
                    {
                        auto viz_msg = cv_bridge::CvImage(pair.header, "bgr8", viz_image).toImageMsg();
                        track_viz_pub_->publish(*viz_msg);
                    }
                }
            }
        }

        void DlVinsComponent::publishOdometry(const std_msgs::msg::Header &header)
        {
            if (!estimator_ || !estimator_->isInitialized())
            {
                return;
            }

            Eigen::Vector3d P, V;
            Eigen::Quaterniond Q;
            double t;
            estimator_->getLatestState(P, Q, V, t);

            // Publish odometry
            if (odom_pub_->get_subscription_count() > 0)
            {
                nav_msgs::msg::Odometry odom_msg;
                odom_msg.header = header;
                odom_msg.header.frame_id = odom_frame_id_;
                odom_msg.child_frame_id = "body";

                odom_msg.pose.pose.position.x = P.x();
                odom_msg.pose.pose.position.y = P.y();
                odom_msg.pose.pose.position.z = P.z();
                odom_msg.pose.pose.orientation.w = Q.w();
                odom_msg.pose.pose.orientation.x = Q.x();
                odom_msg.pose.pose.orientation.y = Q.y();
                odom_msg.pose.pose.orientation.z = Q.z();

                odom_msg.twist.twist.linear.x = V.x();
                odom_msg.twist.twist.linear.y = V.y();
                odom_msg.twist.twist.linear.z = V.z();

                odom_pub_->publish(odom_msg);
            }
        }

        void DlVinsComponent::publishPath(const std_msgs::msg::Header &header)
        {
            if (!estimator_ || !estimator_->isInitialized())
                return;

            Eigen::Vector3d P, V;
            Eigen::Quaterniond Q;
            double t;
            estimator_->getLatestState(P, Q, V, t);

            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header = header;
            pose_msg.header.frame_id = odom_frame_id_;
            pose_msg.pose.position.x = P.x();
            pose_msg.pose.position.y = P.y();
            pose_msg.pose.position.z = P.z();
            pose_msg.pose.orientation.w = Q.w();
            pose_msg.pose.orientation.x = Q.x();
            pose_msg.pose.orientation.y = Q.y();
            pose_msg.pose.orientation.z = Q.z();

            path_msg_.header = header;
            path_msg_.header.frame_id = odom_frame_id_;
            path_msg_.poses.push_back(pose_msg);

            if (path_pub_->get_subscription_count() > 0)
                path_pub_->publish(path_msg_);
        }

        void DlVinsComponent::publishPointCloud(const std_msgs::msg::Header &header)
        {
            if (!point_cloud_pub_ || point_cloud_pub_->get_subscription_count() == 0)
                return;
            if (!estimator_)
                return;

            std::vector<Eigen::Vector3d> landmarks;
            estimator_->getActiveLandmarks(landmarks);

            sensor_msgs::msg::PointCloud2 cloud;
            cloud.header = header;
            cloud.header.frame_id = odom_frame_id_;
            cloud.height = 1;
            cloud.width = static_cast<uint32_t>(landmarks.size());
            cloud.is_dense = true;
            cloud.is_bigendian = false;

            cloud.fields.resize(3);
            const char *names[3] = {"x", "y", "z"};
            for (int i = 0; i < 3; ++i)
            {
                cloud.fields[i].name = names[i];
                cloud.fields[i].offset = static_cast<uint32_t>(i * sizeof(float));
                cloud.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
                cloud.fields[i].count = 1;
            }
            cloud.point_step = 3 * sizeof(float);
            cloud.row_step = cloud.point_step * cloud.width;
            cloud.data.resize(cloud.row_step);

            for (size_t i = 0; i < landmarks.size(); ++i)
            {
                float xyz[3] = {static_cast<float>(landmarks[i].x()),
                                static_cast<float>(landmarks[i].y()),
                                static_cast<float>(landmarks[i].z())};
                std::memcpy(cloud.data.data() + i * cloud.point_step, xyz, sizeof(xyz));
            }

            point_cloud_pub_->publish(cloud);
        }

        void DlVinsComponent::publishKeyframeArtifacts()
        {
            if (!estimator_ || !estimator_->isInitialized())
                return;

            const int kf_idx = Estimator::keyframePublishIndex();
            const double t_kf = estimator_->getHeaderAt(kf_idx);

            Eigen::Vector3d P, tic_v;
            Eigen::Matrix3d R, ric_m;
            estimator_->getKeyframePose(kf_idx, P, R);
            estimator_->getExtrinsic(0, tic_v, ric_m);
            const Eigen::Quaterniond Q(R);
            const Eigen::Quaterniond Q_ic(ric_m);

            std_msgs::msg::Header hdr;
            hdr.stamp = rclcpp::Time(static_cast<int64_t>(t_kf * 1e9));
            hdr.frame_id = odom_frame_id_;

            // /keyframe_pose
            {
                nav_msgs::msg::Odometry msg;
                msg.header = hdr;
                msg.pose.pose.position.x = P.x();
                msg.pose.pose.position.y = P.y();
                msg.pose.pose.position.z = P.z();
                msg.pose.pose.orientation.w = Q.w();
                msg.pose.pose.orientation.x = Q.x();
                msg.pose.pose.orientation.y = Q.y();
                msg.pose.pose.orientation.z = Q.z();
                keyframe_pose_pub_->publish(msg);
            }

            // /extrinsic — republished every keyframe for loop_fusion's cached qic/tic
            {
                nav_msgs::msg::Odometry msg;
                msg.header = hdr;
                msg.pose.pose.position.x = tic_v.x();
                msg.pose.pose.position.y = tic_v.y();
                msg.pose.pose.position.z = tic_v.z();
                msg.pose.pose.orientation.w = Q_ic.w();
                msg.pose.pose.orientation.x = Q_ic.x();
                msg.pose.pose.orientation.y = Q_ic.y();
                msg.pose.pose.orientation.z = Q_ic.z();
                extrinsic_pub_->publish(msg);
            }

            // /keyframe_point — PointCloud2 with 8 typed fields:
            //   x, y, z, norm_x, norm_y, uv_x, uv_y, feature_id
            // All float32 (feature_id rounds harmlessly back to int on consume).
            {
                std::vector<Estimator::KeyframePoint> kps;
                estimator_->getKeyframeLandmarks(kf_idx, kps);

                sensor_msgs::msg::PointCloud2 cloud;
                cloud.header = hdr;
                cloud.height = 1;
                cloud.width = static_cast<uint32_t>(kps.size());
                cloud.is_dense = true;
                cloud.is_bigendian = false;

                constexpr int kNumFields = 8;
                cloud.fields.resize(kNumFields);
                const char *names[kNumFields] = {
                    "x", "y", "z", "norm_x", "norm_y", "uv_x", "uv_y", "feature_id"};
                for (int i = 0; i < kNumFields; ++i)
                {
                    cloud.fields[i].name = names[i];
                    cloud.fields[i].offset = static_cast<uint32_t>(i * sizeof(float));
                    cloud.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
                    cloud.fields[i].count = 1;
                }
                cloud.point_step = kNumFields * sizeof(float);
                cloud.row_step = cloud.point_step * cloud.width;
                cloud.data.resize(cloud.row_step);

                for (size_t i = 0; i < kps.size(); ++i)
                {
                    const auto &kp = kps[i];
                    float buf[kNumFields] = {
                        static_cast<float>(kp.world.x()),
                        static_cast<float>(kp.world.y()),
                        static_cast<float>(kp.world.z()),
                        static_cast<float>(kp.norm.x()),
                        static_cast<float>(kp.norm.y()),
                        static_cast<float>(kp.uv.x()),
                        static_cast<float>(kp.uv.y()),
                        static_cast<float>(kp.feature_id),
                    };
                    std::memcpy(cloud.data.data() + i * cloud.point_step, buf, sizeof(buf));
                }
                keyframe_point_pub_->publish(cloud);
            }

            // /margin_cloud — landmarks leaving the window. PointCloud2 (x,y,z).
            {
                std::vector<Eigen::Vector3d> world;
                estimator_->getMarginalizedLandmarks(world);

                sensor_msgs::msg::PointCloud2 cloud;
                cloud.header = hdr;
                cloud.height = 1;
                cloud.width = static_cast<uint32_t>(world.size());
                cloud.is_dense = true;
                cloud.is_bigendian = false;

                cloud.fields.resize(3);
                const char *names[3] = {"x", "y", "z"};
                for (int i = 0; i < 3; ++i)
                {
                    cloud.fields[i].name = names[i];
                    cloud.fields[i].offset = static_cast<uint32_t>(i * sizeof(float));
                    cloud.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
                    cloud.fields[i].count = 1;
                }
                cloud.point_step = 3 * sizeof(float);
                cloud.row_step = cloud.point_step * cloud.width;
                cloud.data.resize(cloud.row_step);

                for (size_t i = 0; i < world.size(); ++i)
                {
                    float xyz[3] = {static_cast<float>(world[i].x()),
                                    static_cast<float>(world[i].y()),
                                    static_cast<float>(world[i].z())};
                    std::memcpy(cloud.data.data() + i * cloud.point_step, xyz, sizeof(xyz));
                }
                margin_cloud_pub_->publish(cloud);
            }
        }

        void DlVinsComponent::publishIMUOdometry(const std_msgs::msg::Header &header,
                                                 const Eigen::Vector3d &P,
                                                 const Eigen::Quaterniond &Q,
                                                 const Eigen::Vector3d &V)
        {
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header = header;
            odom_msg.header.frame_id = odom_frame_id_;
            odom_msg.child_frame_id = "body";

            odom_msg.pose.pose.position.x = P.x();
            odom_msg.pose.pose.position.y = P.y();
            odom_msg.pose.pose.position.z = P.z();
            odom_msg.pose.pose.orientation.w = Q.w();
            odom_msg.pose.pose.orientation.x = Q.x();
            odom_msg.pose.pose.orientation.y = Q.y();
            odom_msg.pose.pose.orientation.z = Q.z();

            odom_msg.twist.twist.linear.x = V.x();
            odom_msg.twist.twist.linear.y = V.y();
            odom_msg.twist.twist.linear.z = V.z();

            imu_propagate_odom_pub_->publish(odom_msg);
        }

    } // namespace perception
} // namespace uosm

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(uosm::perception::DlVinsComponent)