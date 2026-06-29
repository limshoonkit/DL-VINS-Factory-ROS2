#include "../include/feature_tracker_component.hpp"
#include "../include/utility/metrics_logger.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <Eigen/Dense>

namespace uosm
{
    namespace perception
    {
        FeatureTrackerComponent::FeatureTrackerComponent(const rclcpp::NodeOptions &options) : Node("feature_tracker_component", options)
        {
            RCLCPP_INFO(get_logger(), "**********************************");
            RCLCPP_INFO(get_logger(), " FeatureTracker Component         ");
            RCLCPP_INFO(get_logger(), "**********************************");
            RCLCPP_INFO(get_logger(), " * namespace: %s", get_namespace());
            RCLCPP_INFO(get_logger(), " * node name: %s", get_name());
            RCLCPP_INFO(get_logger(), "**********************************");

            ft_params_.allow_flowback = declare_parameter<bool>("use_flowback", true);
            ft_params_.enable_visualization = declare_parameter<bool>("enable_visualization", false);
            ft_params_.calib_file_cam0 = declare_parameter<std::string>("cam0_calib", "");
            ft_params_.calib_file_cam1 = declare_parameter<std::string>("cam1_calib", "");
            use_stereo_ = !ft_params_.calib_file_cam1.empty();
            ft_params_.col = declare_parameter<int>("image_width", 640U);
            ft_params_.row = declare_parameter<int>("image_height", 480U);
            image_encoding_ = declare_parameter<std::string>("image_encoding", "");

            // Feature detection parameters
            ft_params_.min_feature_distance = declare_parameter<int>("min_feature_distance", 10U);
            ft_params_.max_tracked_keypoints = declare_parameter<int>("max_tracked_keypoints", 512U);

            // Optical flow parameters
            ft_params_.optflow_max_iterations = declare_parameter<int>("optflow_max_iterations", 30U);
            ft_params_.optflow_pyramid_levels = declare_parameter<int>("optflow_pyramid_levels", 3U);
            ft_params_.optflow_window_dim = declare_parameter<int>("optflow_window_dim", 21U);
            ft_params_.optflow_pyramid_scale = declare_parameter<float>("optflow_pyramid_scale", 0.5f);
            ft_params_.optflow_epsilon = declare_parameter<float>("optflow_epsilon", 0.01f);

            // Feature extraction and optical flow backend
            ft_params_.feature_extraction_method = static_cast<FeatureTracker::FeatureExtractionMethod>(
                declare_parameter<int>("feature_extraction_method", 0));
            ft_params_.optflow_device = static_cast<FeatureTracker::OpticalFlowBackend>(
                declare_parameter<int>("optflow_device", 0));
            ft_params_.weights_folder = declare_parameter<std::string>("weights_folder", "");
            ft_params_.use_descriptor_matcher = declare_parameter<bool>("use_descriptor_matcher", false);
            ft_params_.matcher_type = declare_parameter<std::string>("matcher_type", "lightglue");
            ft_params_.matcher_score_threshold =
                static_cast<float>(declare_parameter<double>("matcher_score_threshold", 0.0f));
            ft_params_.profile_trt_inference = declare_parameter<bool>("profile_trt_inference", false);
            ft_params_.fisheye = declare_parameter<bool>("fisheye", false);
            ft_params_.fisheye_mask = declare_parameter<std::string>("fisheye_mask", "");
            enable_csv_logging_ = declare_parameter<bool>("enable_csv_logging", false);
            log_folder_ = declare_parameter<std::string>("log_folder", "./tmp");

            if (use_stereo_)
            {
                ft_params_.stereo_window_dim = declare_parameter<int>("stereo_window_dim", 21U);
                ft_params_.stereo_pyramid_levels = declare_parameter<int>("stereo_pyramid_levels", 3U);

                const std::vector<double> default_identity = {
                    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
                body_T_cam0_ = declare_parameter<std::vector<double>>("body_T_cam0", default_identity);
                body_T_cam1_ = declare_parameter<std::vector<double>>("body_T_cam1", default_identity);
            }

            const char *extraction_methods[] = {"GFTT_CPU", "GFTT_CUDA", "ALIKED", "SuperPoint", "RACO", "XFeat"};
            const char *optflow_devices[] = {"CPU", "CUDA"}; // TODO: Add VPI

            RCLCPP_INFO(get_logger(), "FeatureTrackerComponent parameters:");
            RCLCPP_INFO(get_logger(), " - use_flowback: %s", ft_params_.allow_flowback ? "true" : "false");
            RCLCPP_INFO(get_logger(), " - enable_visualization: %s", ft_params_.enable_visualization ? "true" : "false");
            RCLCPP_INFO(get_logger(), " - cam0_calib: %s", ft_params_.calib_file_cam0.c_str());
            RCLCPP_INFO(get_logger(), " - cam1_calib: %s", ft_params_.calib_file_cam1.c_str());
            RCLCPP_INFO(get_logger(), " - image_width: %d", ft_params_.col);
            RCLCPP_INFO(get_logger(), " - image_height: %d", ft_params_.row);
            RCLCPP_INFO(get_logger(), " - image_encoding: %s", image_encoding_.c_str());
            RCLCPP_INFO(get_logger(), " - min_feature_distance: %d", ft_params_.min_feature_distance);
            RCLCPP_INFO(get_logger(), " - max_tracked_keypoints: %d", ft_params_.max_tracked_keypoints);
            RCLCPP_INFO(get_logger(), " - optflow_max_iterations: %d", ft_params_.optflow_max_iterations);
            RCLCPP_INFO(get_logger(), " - optflow_pyramid_levels: %d", ft_params_.optflow_pyramid_levels);
            RCLCPP_INFO(get_logger(), " - optflow_window_dim: %d", ft_params_.optflow_window_dim);
            RCLCPP_INFO(get_logger(), " - optflow_pyramid_scale: %.2f", ft_params_.optflow_pyramid_scale);
            RCLCPP_INFO(get_logger(), " - optflow_epsilon: %.3f", ft_params_.optflow_epsilon);
            RCLCPP_INFO(get_logger(), " - feature_extraction_method: %s",
                        extraction_methods[static_cast<int>(ft_params_.feature_extraction_method)]);
            RCLCPP_INFO(get_logger(), " - optflow_device: %s",
                        optflow_devices[static_cast<int>(ft_params_.optflow_device)]);
            RCLCPP_INFO(get_logger(), " - weights_folder: %s", ft_params_.weights_folder.c_str());
            RCLCPP_INFO(get_logger(), " - use_descriptor_matcher: %s", ft_params_.use_descriptor_matcher ? "true" : "false");
            RCLCPP_INFO(get_logger(), " - matcher_type: %s", ft_params_.matcher_type.c_str());
            RCLCPP_INFO(get_logger(), " - matcher_score_threshold: %.2f", ft_params_.matcher_score_threshold);
            if (use_stereo_)
            {
                RCLCPP_INFO(get_logger(), " - stereo_window_dim: %d", ft_params_.stereo_window_dim);
                RCLCPP_INFO(get_logger(), " - stereo_pyramid_levels: %d", ft_params_.stereo_pyramid_levels);
            }
            RCLCPP_INFO(get_logger(), " - profile_trt_inference: %s", ft_params_.profile_trt_inference ? "true" : "false");
            RCLCPP_INFO(get_logger(), " - fisheye: %s", ft_params_.fisheye ? "true" : "false");
            RCLCPP_INFO(get_logger(), " - fisheye_mask: %s", ft_params_.fisheye_mask.c_str());
            RCLCPP_INFO(get_logger(), " - enable_csv_logging: %s", enable_csv_logging_ ? "true" : "false");
            RCLCPP_INFO(get_logger(), " - log_folder: %s", log_folder_.c_str());

            std::chrono::milliseconds init_msec(static_cast<int>(10.0));
            init_timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(init_msec),
                                            std::bind(&FeatureTrackerComponent::init, this));
        }

        FeatureTrackerComponent::~FeatureTrackerComponent()
        {
            RCLCPP_WARN(get_logger(), "FeatureTrackerComponent shutting down.");
            {
                std::lock_guard<std::mutex> lk(pending_mtx_);
                stop_ = true;
            }
            pending_cv_.notify_all();
            if (worker_.joinable())
            {
                worker_.join();
            }
            uosm::utility::MetricsLogger::getInstance().shutdown();
        }

        void FeatureTrackerComponent::init()
        {
            if (init_timer_)
            {
                init_timer_->cancel();
                init_timer_.reset();
            }

            // Get package share directory for resolving relative paths
            std::string package_share_dir;
            try
            {
                package_share_dir = ament_index_cpp::get_package_share_directory("dl_vins");
            }
            catch (const std::exception &e)
            {
                RCLCPP_WARN(get_logger(), "Could not get package share directory: %s", e.what());
            }

            // Helper to resolve relative paths
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

            // Resolve relative paths in-place; use_stereo_ already set in ctor and
            // resolution doesn't change emptiness.
            ft_params_.calib_file_cam0 = resolvePath(ft_params_.calib_file_cam0);
            ft_params_.calib_file_cam1 = resolvePath(ft_params_.calib_file_cam1);
            ft_params_.weights_folder = resolvePath(ft_params_.weights_folder);
            ft_params_.fisheye_mask = resolvePath(ft_params_.fisheye_mask);

            RCLCPP_INFO(get_logger(), "Resolved paths:");
            RCLCPP_INFO(get_logger(), "  cam0_calib: %s", ft_params_.calib_file_cam0.c_str());
            RCLCPP_INFO(get_logger(), "  cam1_calib: %s", ft_params_.calib_file_cam1.c_str());
            RCLCPP_INFO(get_logger(), "  weights_folder: %s", ft_params_.weights_folder.c_str());
            RCLCPP_INFO(get_logger(), "  fisheye_mask: %s", ft_params_.fisheye_mask.c_str());

            uosm::utility::MetricsConfig metrics_config;
            metrics_config.enable_csv_logging = enable_csv_logging_;
            metrics_config.log_folder = log_folder_;
            uosm::utility::MetricsLogger::getInstance().configure(metrics_config);

            feature_tracker_ = std::make_unique<FeatureTracker>(ft_params_);

            // Derive baseline + full cam0<-cam1 extrinsic from body_T_cam{0,1}; needed by
            // the descriptor stereo gate. Stereo-only.
            if (use_stereo_ && body_T_cam0_.size() == 16 && body_T_cam1_.size() == 16)
            {
                const Eigen::Matrix4d T_body_cam0 =
                    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(body_T_cam0_.data());
                const Eigen::Matrix4d T_body_cam1 =
                    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(body_T_cam1_.data());

                const Eigen::Matrix3d ric0 = T_body_cam0.block<3, 3>(0, 0);
                const Eigen::Vector3d tic0 = T_body_cam0.block<3, 1>(0, 3);
                const Eigen::Matrix3d ric1 = T_body_cam1.block<3, 3>(0, 0);
                const Eigen::Vector3d tic1 = T_body_cam1.block<3, 1>(0, 3);

                const double baseline = (tic1 - tic0).norm();
                if (baseline > 1e-6)
                {
                    feature_tracker_->setStereoBaseline(static_cast<float>(baseline));
                    const Eigen::Matrix3d R_cam0_cam1 = ric0.transpose() * ric1;
                    const Eigen::Vector3d t_cam0_cam1 = ric0.transpose() * (tic1 - tic0);
                    feature_tracker_->setStereoExtrinsics(R_cam0_cam1, t_cam0_cam1);
                }
            }
            else if (use_stereo_)
            {
                RCLCPP_WARN(get_logger(),
                            "body_T_cam0/body_T_cam1 must each be 16-element row-major; "
                            "descriptor stereo gate will run without extrinsics (all "
                            "stereo candidates will be rejected).");
            }

            RCLCPP_INFO(get_logger(), "FeatureTrackerComponent initialized.");

            // Create visualization publisher if enabled
            if (ft_params_.enable_visualization)
            {
                track_viz_pub_ = create_publisher<sensor_msgs::msg::Image>("image_track", 10);
            }

            // Create feature observations publisher
            feature_pub_ = create_publisher<dl_vins::msg::FeatureObservations>("feature_observations", 10);
            descriptor_pub_ = create_publisher<dl_vins::msg::FrameDescriptors>("frame_descriptors", 10);

            try
            {
                if (use_stereo_)
                {
                    image0_sub_.subscribe(this, IMAGE0);
                    image1_sub_.subscribe(this, IMAGE1);

                    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                        SyncPolicy(10), image0_sub_, image1_sub_);
                    sync_->setMaxIntervalDuration(rclcpp::Duration(std::chrono::milliseconds(5)));
                    sync_->registerCallback(&FeatureTrackerComponent::imagesCallback, this);
                }
                else
                {
                    image_mono_sub_ = create_subscription<sensor_msgs::msg::Image>(
                        IMAGE0,
                        rclcpp::SensorDataQoS(),
                        std::bind(&FeatureTrackerComponent::imageCallback, this, std::placeholders::_1));
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(get_logger(), "Exception during initialization: %s", e.what());
                rclcpp::shutdown();
            }

            worker_ = std::thread(&FeatureTrackerComponent::workerLoop, this);
        }

        void FeatureTrackerComponent::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg)
        {
            try
            {
                cv_bridge::CvImageConstPtr image0_cv;
                if (image_encoding_.empty() || image0_msg->encoding == image_encoding_)
                {
                    image0_cv = cv_bridge::toCvShare(image0_msg, image_encoding_.empty() ? image0_msg->encoding : image_encoding_);
                }
                else
                {
                    image0_cv = cv_bridge::toCvCopy(image0_msg, image_encoding_);
                }

                if (image0_cv->image.empty())
                {
                    RCLCPP_WARN(get_logger(), "Received empty image, skipping frame.");
                    return;
                }

                PendingFrame frame;
                frame.timestamp = image0_msg->header.stamp.sec + image0_msg->header.stamp.nanosec * 1e-9;
                frame.img0 = image0_cv->image.clone();
                // frame.img1 left as empty cv::Mat for mono
                frame.header = image0_msg->header;

                {
                    std::lock_guard<std::mutex> lk(pending_mtx_);
                    if (pending_.has_value())
                    {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                             "Frontend frame drop: previous frame not yet processed (dropped ts=%.6f)",
                                             pending_->timestamp);
                    }
                    pending_ = std::move(frame);
                }
                pending_cv_.notify_one();
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

        void FeatureTrackerComponent::imagesCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg,
                                                     const sensor_msgs::msg::Image::ConstSharedPtr &image1_msg)
        {
            try
            {
                cv_bridge::CvImageConstPtr image0_cv, image1_cv;
                if (image_encoding_.empty() || image0_msg->encoding == image_encoding_)
                {
                    image0_cv = cv_bridge::toCvShare(image0_msg, image_encoding_.empty() ? image0_msg->encoding : image_encoding_);
                    image1_cv = cv_bridge::toCvShare(image1_msg, image_encoding_.empty() ? image1_msg->encoding : image_encoding_);
                }
                else
                {
                    image0_cv = cv_bridge::toCvCopy(image0_msg, image_encoding_);
                    image1_cv = cv_bridge::toCvCopy(image1_msg, image_encoding_);
                }

                if (image0_cv->image.empty() || image1_cv->image.empty())
                {
                    RCLCPP_WARN(get_logger(), "Received empty image(s), skipping frame.");
                    return;
                }

                PendingFrame frame;
                frame.timestamp = image0_msg->header.stamp.sec + image0_msg->header.stamp.nanosec * 1e-9;
                frame.img0 = image0_cv->image.clone();
                frame.img1 = image1_cv->image.clone();
                frame.header = image0_msg->header;

                {
                    std::lock_guard<std::mutex> lk(pending_mtx_);
                    if (pending_.has_value())
                    {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                             "Frontend frame drop: previous frame not yet processed (dropped ts=%.6f)",
                                             pending_->timestamp);
                    }
                    pending_ = std::move(frame);
                }
                pending_cv_.notify_one();
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

        void FeatureTrackerComponent::workerLoop()
        {
            while (true)
            {
                PendingFrame frame;
                {
                    std::unique_lock<std::mutex> lk(pending_mtx_);
                    pending_cv_.wait(lk, [this]
                                     { return stop_ || pending_.has_value(); });
                    if (stop_ && !pending_.has_value())
                    {
                        return;
                    }
                    frame = std::move(*pending_);
                    pending_.reset();
                }
                try
                {
                    processFrame(frame);
                }
                catch (const std::exception &e)
                {
                    RCLCPP_ERROR(get_logger(), "Exception in workerLoop processFrame: %s", e.what());
                }
            }
        }

        void FeatureTrackerComponent::processFrame(const PendingFrame &frame)
        {
            ++frame_count_;
            auto observations = feature_tracker_->trackImage(frame.timestamp, frame.img0, frame.img1);

            if (feature_pub_->get_subscription_count() > 0 && !observations.empty())
            {
                auto feature_msg = dl_vins::msg::FeatureObservations();
                feature_msg.header = frame.header;
                size_t total_features = observations.size();

                feature_msg.num_features = total_features;
                feature_msg.feature_ids.reserve(total_features);
                feature_msg.points_left_x.reserve(total_features);
                feature_msg.points_left_y.reserve(total_features);
                feature_msg.points_right_x.reserve(total_features);
                feature_msg.points_right_y.reserve(total_features);
                feature_msg.pixel_left_x.reserve(total_features);
                feature_msg.pixel_left_y.reserve(total_features);
                feature_msg.pixel_right_x.reserve(total_features);
                feature_msg.pixel_right_y.reserve(total_features);
                feature_msg.velocities_x.reserve(total_features);
                feature_msg.velocities_y.reserve(total_features);

                for (const auto &[feature_id, camera_obs_vec] : observations)
                {
                    Observation left_obs;
                    Observation right_obs;
                    bool has_left = false;
                    bool has_right = false;

                    for (const auto &[camera_id, obs] : camera_obs_vec)
                    {
                        if (camera_id == 0)
                        {
                            left_obs = obs;
                            has_left = true;
                        }
                        else if (camera_id == 1)
                        {
                            right_obs = obs;
                            has_right = true;
                        }
                    }

                    if (has_left)
                    {
                        feature_msg.feature_ids.push_back(feature_id);
                        feature_msg.points_left_x.push_back(left_obs.point_c.x());
                        feature_msg.points_left_y.push_back(left_obs.point_c.y());
                        feature_msg.pixel_left_x.push_back(left_obs.uv.x());
                        feature_msg.pixel_left_y.push_back(left_obs.uv.y());
                        feature_msg.velocities_x.push_back(left_obs.velocity.x());
                        feature_msg.velocities_y.push_back(left_obs.velocity.y());

                        if (has_right)
                        {
                            feature_msg.points_right_x.push_back(right_obs.point_c.x());
                            feature_msg.points_right_y.push_back(right_obs.point_c.y());
                            feature_msg.pixel_right_x.push_back(right_obs.uv.x());
                            feature_msg.pixel_right_y.push_back(right_obs.uv.y());
                        }
                        else
                        {
                            feature_msg.points_right_x.push_back(0.0);
                            feature_msg.points_right_y.push_back(0.0);
                            feature_msg.pixel_right_x.push_back(0.0);
                            feature_msg.pixel_right_y.push_back(0.0);
                        }
                    }
                }

                feature_pub_->publish(feature_msg);
            }

            if (descriptor_pub_ && descriptor_pub_->get_subscription_count() > 0)
            {
                const auto &desc = feature_tracker_->getCurrentDescriptors();
                const auto &desc_ids = feature_tracker_->getDescriptorIds();
                if (!desc.empty() && !desc_ids.empty())
                {
                    auto desc_msg = dl_vins::msg::FrameDescriptors();
                    desc_msg.header = frame.header;
                    desc_msg.descriptor_dim = static_cast<uint32_t>(desc.cols);
                    desc_msg.feature_ids.assign(desc_ids.begin(), desc_ids.end());
                    desc_msg.descriptors.resize(desc.rows * desc.cols);
                    std::memcpy(desc_msg.descriptors.data(), desc.data,
                                desc.rows * desc.cols * sizeof(float));
                    descriptor_pub_->publish(desc_msg);
                }
            }

            if (ft_params_.enable_visualization && track_viz_pub_->get_subscription_count() > 0)
            {
                cv::Mat viz_image = feature_tracker_->getTrackVisualization();
                if (viz_image.empty())
                {
                    RCLCPP_WARN(get_logger(), "Received empty visualization image, skipping frame.");
                    return;
                }

                std_msgs::msg::Header header;
                header.stamp = frame.header.stamp;
                header.frame_id = frame.header.frame_id;

                auto viz_msg = cv_bridge::CvImage(header, "bgr8", viz_image).toImageMsg();
                track_viz_pub_->publish(*viz_msg);
            }
        }
    } // namespace perception
} // namespace uosm

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(uosm::perception::FeatureTrackerComponent)