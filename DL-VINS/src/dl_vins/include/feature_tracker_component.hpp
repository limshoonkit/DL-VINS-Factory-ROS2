#ifndef FEATURE_TRACKER_COMPONENT_HPP_
#define FEATURE_TRACKER_COMPONENT_HPP_

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

#include "feature/feature_tracker.hpp"
#include "dl_vins/msg/feature_observations.hpp"
#include "dl_vins/msg/frame_descriptors.hpp"

namespace uosm
{
    namespace perception
    {
        class FeatureTrackerComponent : public rclcpp::Node
        {
        public:
            explicit FeatureTrackerComponent(const rclcpp::NodeOptions &options);
            ~FeatureTrackerComponent();

        private:
            std::unique_ptr<FeatureTracker> feature_tracker_;

            // Constants
            const std::string IMAGE0 = "image0";
            const std::string IMAGE1 = "image1";

            // State
            uint64_t frame_count_ = 0UL;
            bool use_stereo_ = true;

            FeatureTracker::FeatureTrackerParams ft_params_;
            std::string image_encoding_;
            bool enable_csv_logging_;
            std::string log_folder_;
            std::vector<double> body_T_cam0_;
            std::vector<double> body_T_cam1_;

            // ROS2
            using SyncPolicy = message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
            message_filters::Subscriber<sensor_msgs::msg::Image> image0_sub_;
            message_filters::Subscriber<sensor_msgs::msg::Image> image1_sub_;
            std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
            rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_mono_sub_;

            rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr track_viz_pub_;
            rclcpp::Publisher<dl_vins::msg::FeatureObservations>::SharedPtr feature_pub_;
            rclcpp::Publisher<dl_vins::msg::FrameDescriptors>::SharedPtr descriptor_pub_;

            // Timers
            rclcpp::TimerBase::SharedPtr init_timer_;

            // If a new frame arrives while one is pending,
            // the older one is dropped and a throttled warning is logged.
            struct PendingFrame
            {
                double timestamp;
                cv::Mat img0;
                cv::Mat img1; // empty for mono
                std_msgs::msg::Header header;
            };
            std::optional<PendingFrame> pending_;
            std::mutex pending_mtx_;
            std::condition_variable pending_cv_;
            std::thread worker_;
            bool stop_ = false;

            // Callbacks
            void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg);
            void imagesCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image0_msg, const sensor_msgs::msg::Image::ConstSharedPtr &image1_msg);

            // Helpers
            void init();
            void workerLoop();
            void processFrame(const PendingFrame &frame);
        };

    } // namespace perception
} // namespace uosm

#endif // FEATURE_TRACKER_COMPONENT_HPP_