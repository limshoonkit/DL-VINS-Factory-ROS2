#ifndef DL_VINS_COMPONENT_HPP_
#define DL_VINS_COMPONENT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#include <condition_variable>
#include <queue>
#include <mutex>
#include <stop_token>
#include <thread>

#include "estimator/estimator.hpp"
#include "feature/feature_manager.hpp"
#include "feature/feature_tracker.hpp"
#include "utility/metrics_logger.hpp"
#include "dl_vins/msg/frame_descriptors.hpp"

namespace uosm
{
    namespace perception
    {
        class DlVinsComponent : public rclcpp::Node
        {
        public:
            struct StampedImagePair
            {
                double timestamp;
                cv::Mat img0;
                cv::Mat img1;
                std_msgs::msg::Header header;
            };

            explicit DlVinsComponent(const rclcpp::NodeOptions &options);
            virtual ~DlVinsComponent();

        private:
            EstimatorConfig estimator_config_;
            FeatureTracker::FeatureTrackerParams tracker_config_;
            std::string image_encoding_;
            std::string odom_frame_id_;
            std::string body_frame_id_;
            bool publish_tf_{true};
            double norm_focal_length_{0.0};

            // TF tree
            std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
            std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

            // IMU subscriber
            rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

            // Stereo image subscribers
            using SyncPolicy = message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
            message_filters::Subscriber<sensor_msgs::msg::Image> image0_sub_;
            message_filters::Subscriber<sensor_msgs::msg::Image> image1_sub_;
            std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> image_sync_;
            rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_mono_sub_;

            // Publishers
            rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr imu_propagate_odom_pub_;
            rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
            rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;
            rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr track_viz_pub_;
            rclcpp::Publisher<dl_vins::msg::FrameDescriptors>::SharedPtr descriptor_pub_;
            rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr keyframe_pose_pub_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr keyframe_point_pub_;
            rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr extrinsic_pub_;
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr margin_cloud_pub_;
            nav_msgs::msg::Path path_msg_;

            // Callbacks
            void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &imu_msg);
            void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg);
            void imagesCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg,
                                const sensor_msgs::msg::Image::ConstSharedPtr &image1_msg);

            // Timers
            rclcpp::TimerBase::SharedPtr init_timer_;

            // Image buffer for tracking thread
            std::mutex image_buf_mutex_;
            std::condition_variable_any image_buf_cv_;
            std::queue<StampedImagePair> image_buf_;

            // Tracking thread
            std::jthread tracking_thread_;
            void trackingLoop(std::stop_token st);

            // State
            int input_image_count_{0};

            uint64_t last_published_keyframe_seq_{0};
            uint64_t last_published_backend_seq_{0};

            // Core components
            std::unique_ptr<Estimator> estimator_;
            std::unique_ptr<FeatureTracker> feature_tracker_;

            // Helpers
            void init();
            void publishOdometry(const std_msgs::msg::Header &header);
            void publishPath(const std_msgs::msg::Header &header);
            void publishPointCloud(const std_msgs::msg::Header &header);
            void publishIMUOdometry(const std_msgs::msg::Header &header,
                                    const Eigen::Vector3d &P,
                                    const Eigen::Quaterniond &Q,
                                    const Eigen::Vector3d &V);
            void publishKeyframeArtifacts();
            void publishStaticExtrinsicTF(const Eigen::Vector3d &tic0, const Eigen::Quaterniond &ric0,
                                          const Eigen::Vector3d &tic1, const Eigen::Quaterniond &ric1);

            // Metrics / logging
            bool enable_csv_logging_ = false;
            std::string log_folder_;
            bool use_stereo_ = true;

            // Extrinsic calibration
            std::vector<double> body_T_cam0_;
            std::vector<double> body_T_cam1_;
        };

    } // namespace perception
} // namespace uosm

#endif // DL_VINS_COMPONENT_HPP_