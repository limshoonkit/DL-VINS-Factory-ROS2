#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <dl_vins/msg/frame_descriptors.hpp>

class KeyFrame;

namespace uosm::loop_fusion
{
    class DinoVpr; // AnyLoc DINOv2 + VLAD global-VPR head
}

namespace uosm::perception
{

    class LoopFusionComponent : public rclcpp::Node
    {
    public:
        explicit LoopFusionComponent(const rclcpp::NodeOptions &options);
        ~LoopFusionComponent() override;

    private:
        void loadConfig(const std::string &config_file,
                        const std::string &cam0_calib_path_override);
        void newSequence();

        void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
        void pointCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
        void marginPointCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
        void poseCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
        void vioCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
        void extrinsicCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
        void descriptorCallback(const dl_vins::msg::FrameDescriptors::ConstSharedPtr msg);

        void processLoop();
        void loadDinoVpr(const cv::FileStorage &fs, const std::string &pkg_share);
        void computeDinoGlobalDesc(KeyFrame *keyframe, const cv::Mat &image);

        void onSaveRequest(
            const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
            std::shared_ptr<std_srvs::srv::Trigger::Response> res);
        void onNewSequenceRequest(
            const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
            std::shared_ptr<std_srvs::srv::Trigger::Response> res);

        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_vio_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pose_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_extrinsic_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_point_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_margin_;
        rclcpp::Subscription<dl_vins::msg::FrameDescriptors>::SharedPtr sub_descriptor_;

        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_save_;
        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_new_seq_;

        std::thread process_thread_;
        std::atomic<bool> running_{false};
        std::string IMAGE_TOPIC_;

        // DL mode: accumulated learned descriptors keyed by feature id
        std::unordered_map<int, std::vector<float>> id_to_descriptor_;
        std::mutex m_desc_;
        std::unique_ptr<uosm::loop_fusion::DinoVpr> dino_vpr_;
    };

} // namespace uosm::perception
