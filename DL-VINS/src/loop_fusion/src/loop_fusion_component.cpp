#include "loop_fusion/loop_fusion_component.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "loop_fusion/dino_vpr.hpp"
#include "loop_fusion/keyframe.hpp"
#include "loop_fusion/parameters.hpp"
#include "loop_fusion/pose_graph.hpp"
#include "loop_fusion/utility/camera_pose_visualization.hpp"
#include "loop_fusion/utility/loop_metrics_logger.hpp"

#define SKIP_FIRST_CNT 10

// ---- globals declared as extern in parameters.h ----------------------------
camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match_img;
int VISUALIZATION_SHIFT_X = 0;
int VISUALIZATION_SHIFT_Y = 0;
std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
int ROW = 0;
int COL = 0;
int DEBUG_IMAGE = 0;
double FOCAL_LENGTH_PX = 460.0; // recomputed from m_camera at config load

// Loop closure tunables
double max_focallength = 460.0;
double MIN_SCORE = 0.03;
double PNP_INFLATION = 10.0;
int RECALL_IGNORE_RECENT_COUNT = 50;
double MAX_THETA_DIFF = 30.0;
double MAX_POS_DIFF = 20.0;
int MIN_LOOP_NUM = 25;

// 4-DoF pose-graph optimization budget.
int POSE_GRAPH_OPT_WINDOW = 0; // 0 = unbounded
double POSE_GRAPH_MAX_SOLVER_TIME = 5.0;
int POSE_GRAPH_MAX_ITERATIONS = 20;
int POSE_GRAPH_NUM_THREADS = 1;

// AnyLoc global-VPR loop closure (DINOv2 ViT-S + VLAD).
bool USE_DL_LOOP = false;
std::string DESCRIPTOR_TOPIC = "frame_descriptors";
double DL_LOOP_SIM_THRESH = 0.60;
double DL_RATIO_TEST = 0.85;

// ---- TU-local globals ---------------
namespace
{
    std::queue<sensor_msgs::msg::Image::ConstSharedPtr> image_buf;
    std::queue<sensor_msgs::msg::PointCloud2::ConstSharedPtr> point_buf;
    std::queue<nav_msgs::msg::Odometry::ConstSharedPtr> pose_buf;
    std::mutex m_buf;
    std::mutex m_process;
    int frame_index = 0;
    int sequence = 1;
    PoseGraph posegraph;
    int skip_first_cnt = 0;
    int SKIP_CNT = 0;
    int skip_cnt = 0;
    bool start_flag = false;
    double SKIP_DIS = 0;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry_rect;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_point_cloud;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_margin_cloud;

    Eigen::Vector3d last_t(-100, -100, -100);
    double last_image_time = -1;
} // namespace

namespace uosm::perception
{

    LoopFusionComponent::LoopFusionComponent(const rclcpp::NodeOptions &options)
        : Node("loop_fusion", options)
    {
        const std::string config_file = declare_parameter<std::string>("config_file", "");
        if (config_file.empty())
        {
            RCLCPP_FATAL(get_logger(), "loop_fusion: parameter 'config_file' is required");
            throw std::runtime_error("loop_fusion: missing 'config_file' parameter");
        }
        const std::string cam0_calib_path_param =
            declare_parameter<std::string>("cam0_calib_path", "");
        loadConfig(config_file, cam0_calib_path_param);

        posegraph.registerPub(this);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(2000));
        sub_vio_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", qos,
            [this](const nav_msgs::msg::Odometry::ConstSharedPtr m)
            { vioCallback(m); });
        sub_image_ = create_subscription<sensor_msgs::msg::Image>(
            IMAGE_TOPIC_, qos,
            [this](const sensor_msgs::msg::Image::ConstSharedPtr m)
            { imageCallback(m); });
        sub_pose_ = create_subscription<nav_msgs::msg::Odometry>(
            "/keyframe_pose", qos,
            [this](const nav_msgs::msg::Odometry::ConstSharedPtr m)
            { poseCallback(m); });
        sub_extrinsic_ = create_subscription<nav_msgs::msg::Odometry>(
            "/extrinsic", qos,
            [this](const nav_msgs::msg::Odometry::ConstSharedPtr m)
            { extrinsicCallback(m); });
        sub_point_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/keyframe_point", qos,
            [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr m)
            { pointCallback(m); });
        sub_margin_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/margin_cloud", qos,
            [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr m)
            { marginPointCallback(m); });
        if (USE_DL_LOOP)
        {
            sub_descriptor_ = create_subscription<dl_vins::msg::FrameDescriptors>(
                DESCRIPTOR_TOPIC, qos,
                [this](const dl_vins::msg::FrameDescriptors::ConstSharedPtr m)
                { descriptorCallback(m); });
        }

        pub_match_img = create_publisher<sensor_msgs::msg::Image>("match_image", 1000);
        pub_point_cloud = create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud_loop_rect", 1000);
        pub_margin_cloud = create_publisher<sensor_msgs::msg::PointCloud2>("margin_cloud_loop_rect", 1000);
        pub_odometry_rect = create_publisher<nav_msgs::msg::Odometry>("odometry_rect", 1000);

        srv_save_ = create_service<std_srvs::srv::Trigger>(
            "~/save_pose_graph",
            [this](
                const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                std::shared_ptr<std_srvs::srv::Trigger::Response> res)
            { onSaveRequest(req, res); });
        srv_new_seq_ = create_service<std_srvs::srv::Trigger>(
            "~/new_sequence",
            [this](
                const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                std::shared_ptr<std_srvs::srv::Trigger::Response> res)
            { onNewSequenceRequest(req, res); });

        running_ = true;
        process_thread_ = std::thread([this]()
                                      { processLoop(); });

        RCLCPP_INFO(get_logger(), "loop_fusion: composable component up; spinning on container");
    }

    LoopFusionComponent::~LoopFusionComponent()
    {
        running_ = false;
        if (process_thread_.joinable())
        {
            process_thread_.join();
        }
    }

    void LoopFusionComponent::loadConfig(const std::string &config_file,
                                         const std::string &cam0_calib_path_override)
    {
        RCLCPP_INFO(get_logger(), "loop_fusion config_file: %s", config_file.c_str());
        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            RCLCPP_FATAL(get_logger(), "Bad config file path: %s", config_file.c_str());
            throw std::runtime_error("loop_fusion: cannot open config_file");
        }

        ROW = static_cast<int>(fs["image_height"]);
        COL = static_cast<int>(fs["image_width"]);

        const std::string pkg_share = ament_index_cpp::get_package_share_directory("loop_fusion");

        std::string cam0Path = cam0_calib_path_override;
        if (cam0Path.empty())
        {
            const auto pn = config_file.find_last_of('/');
            const std::string configPath = config_file.substr(0, pn);
            std::string cam0Calib;
            fs["cam0_calib"] >> cam0Calib;
            cam0Path = configPath + "/" + cam0Calib;
        }
        RCLCPP_INFO(get_logger(), "cam calib path: %s", cam0Path.c_str());
        m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());

        // Derive effective focal length in pixels from the actual camera
        // model parameters
        {
            Eigen::Vector2d p_center, p_off;
            m_camera->spaceToPlane(Eigen::Vector3d(0.0, 0.0, 1.0), p_center);
            m_camera->spaceToPlane(Eigen::Vector3d(0.01, 0.0, 1.0), p_off);
            const double f = (p_off - p_center).norm() / 0.01;
            if (std::isfinite(f) && f > 1.0)
                FOCAL_LENGTH_PX = f;
            max_focallength = FOCAL_LENGTH_PX;
            RCLCPP_INFO(get_logger(),
                        "Derived FOCAL_LENGTH_PX = %.2f px from m_camera", FOCAL_LENGTH_PX);
        }

        fs["image0_topic"] >> IMAGE_TOPIC_;
        fs["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
        fs["save_image"] >> DEBUG_IMAGE;
        int load_previous = static_cast<int>(fs["load_previous_pose_graph"]);

        auto readDouble = [&fs](const char *key, double &out)
        {
            const cv::FileNode n = fs[key];
            if (!n.empty())
                out = static_cast<double>(n);
        };
        auto readInt = [&fs](const char *key, int &out)
        {
            const cv::FileNode n = fs[key];
            if (!n.empty())
                out = static_cast<int>(n);
        };
        readDouble("min_score", MIN_SCORE);
        readDouble("pnp_inflation", PNP_INFLATION);
        readInt("recall_ignore_recent_ct", RECALL_IGNORE_RECENT_COUNT);
        readDouble("max_theta_diff", MAX_THETA_DIFF);
        readDouble("max_pos_diff", MAX_POS_DIFF);
        readInt("min_loop_feat_num", MIN_LOOP_NUM);
        readInt("pose_graph_opt_window", POSE_GRAPH_OPT_WINDOW);
        readDouble("pose_graph_max_solver_time", POSE_GRAPH_MAX_SOLVER_TIME);
        readInt("pose_graph_max_iterations", POSE_GRAPH_MAX_ITERATIONS);
        readInt("pose_graph_num_threads", POSE_GRAPH_NUM_THREADS);
        RCLCPP_INFO(get_logger(),
                    "loop tunables: MIN_SCORE=%.3f PNP_INFLATION=%.2f "
                    "RECALL_IGNORE_RECENT_COUNT=%d MAX_THETA_DIFF=%.2f "
                    "MAX_POS_DIFF=%.2f MIN_LOOP_NUM=%d",
                    MIN_SCORE, PNP_INFLATION, RECALL_IGNORE_RECENT_COUNT,
                    MAX_THETA_DIFF, MAX_POS_DIFF, MIN_LOOP_NUM);
        RCLCPP_INFO(get_logger(),
                    "pose-graph opt: window=%d (0=unbounded) max_solver_time=%.2fs max_iters=%d threads=%d",
                    POSE_GRAPH_OPT_WINDOW, POSE_GRAPH_MAX_SOLVER_TIME, POSE_GRAPH_MAX_ITERATIONS,
                    POSE_GRAPH_NUM_THREADS);

        // ── Loop-closure mode ─────────────────────────────────────────────────
        //   "classic"    -> DBoW2 + BRIEF (GFTT / LK front-ends)
        //   "global_vpr" -> AnyLoc DINOv2 ViT-S + VLAD retrieval, with LightGlue
        //                   local-descriptor verification (DL front-ends)
        {
            std::string mode = "classic";
            const cv::FileNode mode_n = fs["loop_closure_mode"];
            if (!mode_n.empty())
                mode_n >> mode;
            // Launch-driven override (per front-end; see euroc_mono.launch.py).
            const std::string mode_param =
                declare_parameter<std::string>("loop_closure_mode", "");
            if (!mode_param.empty())
                mode = mode_param;
            USE_DL_LOOP = (mode == "global_vpr");

            const cv::FileNode topic_n = fs["descriptor_topic"];
            if (!topic_n.empty())
                topic_n >> DESCRIPTOR_TOPIC;
            const std::string topic_param =
                declare_parameter<std::string>("descriptor_topic", "");
            if (!topic_param.empty())
                DESCRIPTOR_TOPIC = topic_param;
            readDouble("dl_loop_sim_thresh", DL_LOOP_SIM_THRESH);
            readDouble("dino_loop_sim_thresh", DL_LOOP_SIM_THRESH);
            readDouble("dl_ratio_test", DL_RATIO_TEST);

            if (USE_DL_LOOP)
            {
                loadDinoVpr(fs, pkg_share);
            }
            else
            {
                // Classic DBoW2+BRIEF path: load vocab and pattern only here.
                const std::string vocabulary_file = pkg_share + "/support_files/brief_k10L6.bin";
                RCLCPP_INFO(get_logger(), "vocabulary: %s", vocabulary_file.c_str());
                posegraph.loadVocabulary(vocabulary_file);

                BRIEF_PATTERN_FILE = pkg_share + "/support_files/brief_pattern.yml";
                RCLCPP_INFO(get_logger(), "BRIEF_PATTERN_FILE: %s", BRIEF_PATTERN_FILE.c_str());

                RCLCPP_INFO(get_logger(), "loop closure: CLASSIC mode (DBoW2 + BRIEF)");
            }
        }

        // Configure standalone loop-fusion metrics logger.
        {
            uosm::loop_fusion::LoopFusionConfig lf_cfg;
            lf_cfg.enable_csv_logging =
                declare_parameter<bool>("enable_csv_logging", false);
            lf_cfg.log_folder =
                declare_parameter<std::string>("log_folder", "./tmp/dl_vins_logs");
            auto &logger = uosm::loop_fusion::LoopMetricsLogger::getInstance();
            logger.configure(lf_cfg);
        }

        // loop_fusion is built 4-DoF (x, y, z, yaw) only; assume IMU
        posegraph.startOptimization();
        fs.release();

        if (load_previous)
        {
            RCLCPP_INFO(get_logger(), "loading previous pose graph...");
            std::lock_guard<std::mutex> lk(m_process);
            posegraph.loadPoseGraph();
            RCLCPP_INFO(get_logger(), "load pose graph finish");
        }
    }

    void LoopFusionComponent::newSequence()
    {
        RCLCPP_INFO(get_logger(), "new sequence");
        sequence++;
        if (sequence > 5)
        {
            RCLCPP_WARN(get_logger(),
                        "only support 5 sequences since it's boring to copy code for more sequences.");
        }
        posegraph.posegraph_visualization->reset();
        posegraph.publish();
        std::lock_guard<std::mutex> lk(m_buf);
        while (!image_buf.empty())
            image_buf.pop();
        while (!point_buf.empty())
            point_buf.pop();
        while (!pose_buf.empty())
            pose_buf.pop();
    }

    void LoopFusionComponent::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        {
            std::lock_guard<std::mutex> lk(m_buf);
            image_buf.push(msg);
        }
        const double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        if (last_image_time < 0)
        {
            last_image_time = t;
        }
        else if (t - last_image_time > 1.0 || t < last_image_time)
        {
            RCLCPP_WARN(get_logger(), "image discontinue! detect a new sequence!");
            newSequence();
        }
        last_image_time = t;
    }

    namespace
    {
        // Apply (r_drift, t_drift) to every xyz point in an incoming PointCloud2 and
        // repackage as a fresh xyz-only PointCloud2 (visualization only).
        sensor_msgs::msg::PointCloud2 driftCorrectXYZ(const sensor_msgs::msg::PointCloud2 &in)
        {
            sensor_msgs::msg::PointCloud2 out;
            out.header = in.header;
            out.height = 1;
            out.width = in.width * in.height;
            out.is_dense = in.is_dense;
            out.is_bigendian = false;
            out.fields.resize(3);
            const char *names[3] = {"x", "y", "z"};
            for (int i = 0; i < 3; ++i)
            {
                out.fields[i].name = names[i];
                out.fields[i].offset = static_cast<uint32_t>(i * sizeof(float));
                out.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
                out.fields[i].count = 1;
            }
            out.point_step = 3 * sizeof(float);
            out.row_step = out.point_step * out.width;
            out.data.resize(out.row_step);

            sensor_msgs::PointCloud2ConstIterator<float> ix(in, "x"), iy(in, "y"), iz(in, "z");
            for (uint32_t k = 0; k < out.width; ++k, ++ix, ++iy, ++iz)
            {
                Eigen::Vector3d tmp =
                    posegraph.r_drift * Eigen::Vector3d(*ix, *iy, *iz) + posegraph.t_drift;
                float xyz[3] = {static_cast<float>(tmp(0)),
                                static_cast<float>(tmp(1)),
                                static_cast<float>(tmp(2))};
                std::memcpy(out.data.data() + k * out.point_step, xyz, sizeof(xyz));
            }
            return out;
        }
    } // namespace

    void LoopFusionComponent::pointCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
    {
        {
            std::lock_guard<std::mutex> lk(m_buf);
            point_buf.push(msg);
        }
        pub_point_cloud->publish(driftCorrectXYZ(*msg));
    }

    void LoopFusionComponent::marginPointCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
    {
        pub_margin_cloud->publish(driftCorrectXYZ(*msg));
    }

    void LoopFusionComponent::poseCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(m_buf);
        pose_buf.push(msg);
    }

    void LoopFusionComponent::vioCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        Eigen::Vector3d vio_t(
            msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        Eigen::Quaterniond vio_q(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z);

        vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
        vio_q = posegraph.w_r_vio * vio_q;
        vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
        vio_q = posegraph.r_drift * vio_q;

        nav_msgs::msg::Odometry odom;
        odom.header = msg->header;
        odom.header.frame_id = "world";
        odom.pose.pose.position.x = vio_t.x();
        odom.pose.pose.position.y = vio_t.y();
        odom.pose.pose.position.z = vio_t.z();
        odom.pose.pose.orientation.x = vio_q.x();
        odom.pose.pose.orientation.y = vio_q.y();
        odom.pose.pose.orientation.z = vio_q.z();
        odom.pose.pose.orientation.w = vio_q.w();
        pub_odometry_rect->publish(odom);

    }

    void LoopFusionComponent::extrinsicCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(m_process);
        tic = Eigen::Vector3d(
            msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        qic = Eigen::Quaterniond(
                  msg->pose.pose.orientation.w,
                  msg->pose.pose.orientation.x,
                  msg->pose.pose.orientation.y,
                  msg->pose.pose.orientation.z)
                  .toRotationMatrix();
    }

    void LoopFusionComponent::loadDinoVpr(const cv::FileStorage &fs,
                                          const std::string &pkg_share)
    {
        auto resolve = [&pkg_share](std::string name) -> std::string
        {
            if (name.empty())
                return name;
            if (name.front() == '/')
                return name; // absolute path
            return pkg_share + "/support_files/" + name;
        };
        auto readStr = [&](const char *yaml_key, const char *param_key,
                           std::string &out)
        {
            const cv::FileNode n = fs[yaml_key];
            if (!n.empty())
                n >> out;
            const std::string pv = declare_parameter<std::string>(param_key, "");
            if (!pv.empty())
                out = pv;
        };
        auto readIntP = [&](const char *yaml_key, int &out)
        {
            const cv::FileNode n = fs[yaml_key];
            if (!n.empty())
                out = static_cast<int>(n);
        };

        uosm::loop_fusion::DinoVpr::Params dp;
        std::string engine_name, vocab_name;
        readStr("dino_vpr_engine", "dino_vpr_engine", engine_name);
        readStr("dino_vpr_vocab", "dino_vpr_vocab", vocab_name);
        readIntP("dino_vpr_input_h", dp.input_h);
        readIntP("dino_vpr_input_w", dp.input_w);
        readIntP("dino_vpr_embed_dim", dp.embed_dim);

        if (engine_name.empty() || vocab_name.empty())
        {
            RCLCPP_ERROR(get_logger(),
                         "loop closure GLOBAL_VPR: dino_vpr_engine / dino_vpr_vocab "
                         "not set — loop closure disabled");
            return;
        }
        dp.engine_path = resolve(engine_name);
        dp.vocab_path = resolve(vocab_name);

        auto vpr = std::make_unique<uosm::loop_fusion::DinoVpr>(dp);
        if (!vpr->ok())
        {
            RCLCPP_ERROR(get_logger(),
                         "loop closure GLOBAL_VPR: DinoVpr init failed "
                         "(engine=%s vocab=%s) — loop closure disabled",
                         dp.engine_path.c_str(), dp.vocab_path.c_str());
            return;
        }
        dino_vpr_ = std::move(vpr);
        RCLCPP_INFO(get_logger(),
                    "loop closure GLOBAL_VPR: AnyLoc DINOv2+VLAD ready "
                    "(engine=%s %dx%d, VLAD dim=%d, sim_thresh=%.2f ratio=%.2f)",
                    engine_name.c_str(), dp.input_h, dp.input_w,
                    dino_vpr_->vladDim(), DL_LOOP_SIM_THRESH, DL_RATIO_TEST);
    }

    void LoopFusionComponent::computeDinoGlobalDesc(KeyFrame *keyframe,
                                                    const cv::Mat &image)
    {
        if (!dino_vpr_ || keyframe == nullptr)
            return;
        const auto t_vpr_start = std::chrono::high_resolution_clock::now();
        std::vector<float> desc = dino_vpr_->compute(image);
        keyframe->vpr_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - t_vpr_start)
                               .count();
        if (!desc.empty())
            keyframe->dl_global_desc = std::move(desc);
    }

    void LoopFusionComponent::descriptorCallback(
        const dl_vins::msg::FrameDescriptors::ConstSharedPtr msg)
    {
        const int dim = static_cast<int>(msg->descriptor_dim);
        if (dim <= 0 || msg->feature_ids.empty())
            return;
        const size_t n = msg->feature_ids.size();
        if (msg->descriptors.size() < n * static_cast<size_t>(dim))
            return;
        std::lock_guard<std::mutex> lk(m_desc_);
        for (size_t i = 0; i < n; ++i)
        {
            const float *src = msg->descriptors.data() + i * static_cast<size_t>(dim);
            id_to_descriptor_[static_cast<int>(msg->feature_ids[i])] =
                std::vector<float>(src, src + dim);
        }
    }

    void LoopFusionComponent::processLoop()
    {
        using namespace std::chrono_literals;
        while (running_.load() && rclcpp::ok())
        {
            sensor_msgs::msg::Image::ConstSharedPtr image_msg;
            sensor_msgs::msg::PointCloud2::ConstSharedPtr point_msg;
            nav_msgs::msg::Odometry::ConstSharedPtr pose_msg;

            // time-sync: same logic as upstream process()
            m_buf.lock();
            if (!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
            {
                auto t_img_front = [&]
                { return image_buf.front()->header.stamp.sec + image_buf.front()->header.stamp.nanosec * 1e-9; };
                auto t_pt_front = [&]
                { return point_buf.front()->header.stamp.sec + point_buf.front()->header.stamp.nanosec * 1e-9; };
                auto t_pose_front = [&]
                { return pose_buf.front()->header.stamp.sec + pose_buf.front()->header.stamp.nanosec * 1e-9; };
                auto t_img_back = [&]
                { return image_buf.back()->header.stamp.sec + image_buf.back()->header.stamp.nanosec * 1e-9; };
                auto t_pt_back = [&]
                { return point_buf.back()->header.stamp.sec + point_buf.back()->header.stamp.nanosec * 1e-9; };

                if (t_img_front() > t_pose_front())
                {
                    pose_buf.pop();
                }
                else if (t_img_front() > t_pt_front())
                {
                    point_buf.pop();
                }
                else if (t_img_back() >= t_pose_front() && t_pt_back() >= t_pose_front())
                {
                    pose_msg = pose_buf.front();
                    pose_buf.pop();
                    while (!pose_buf.empty())
                        pose_buf.pop();
                    while (t_img_front() < pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * 1e-9)
                        image_buf.pop();
                    image_msg = image_buf.front();
                    image_buf.pop();
                    while (t_pt_front() < pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * 1e-9)
                        point_buf.pop();
                    point_msg = point_buf.front();
                    point_buf.pop();
                }
            }
            m_buf.unlock();

            if (pose_msg)
            {
                if (skip_first_cnt < SKIP_FIRST_CNT)
                {
                    skip_first_cnt++;
                    std::this_thread::sleep_for(5ms);
                    continue;
                }
                if (skip_cnt < SKIP_CNT)
                {
                    skip_cnt++;
                    std::this_thread::sleep_for(5ms);
                    continue;
                }
                skip_cnt = 0;

                cv_bridge::CvImageConstPtr ptr;
                if (image_msg->encoding == "8UC1")
                {
                    sensor_msgs::msg::Image img;
                    img.header = image_msg->header;
                    img.height = image_msg->height;
                    img.width = image_msg->width;
                    img.is_bigendian = image_msg->is_bigendian;
                    img.step = image_msg->step;
                    img.data = image_msg->data;
                    img.encoding = "mono8";
                    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                }
                else
                {
                    ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
                }
                cv::Mat image = ptr->image;

                Eigen::Vector3d T(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
                Eigen::Matrix3d R = Eigen::Quaterniond(
                                        pose_msg->pose.pose.orientation.w,
                                        pose_msg->pose.pose.orientation.x,
                                        pose_msg->pose.pose.orientation.y,
                                        pose_msg->pose.pose.orientation.z)
                                        .toRotationMatrix();

                if ((T - last_t).norm() > SKIP_DIS)
                {
                    std::vector<cv::Point3f> point_3d;
                    std::vector<cv::Point2f> point_2d_uv;
                    std::vector<cv::Point2f> point_2d_normal;
                    std::vector<double> point_id;
                    // PointCloud2 fields (per dl_vins publisher):
                    //   x, y, z, norm_x, norm_y, uv_x, uv_y, feature_id (all float32)
                    sensor_msgs::PointCloud2ConstIterator<float> ix(*point_msg, "x");
                    sensor_msgs::PointCloud2ConstIterator<float> iy(*point_msg, "y");
                    sensor_msgs::PointCloud2ConstIterator<float> iz(*point_msg, "z");
                    sensor_msgs::PointCloud2ConstIterator<float> inx(*point_msg, "norm_x");
                    sensor_msgs::PointCloud2ConstIterator<float> iny(*point_msg, "norm_y");
                    sensor_msgs::PointCloud2ConstIterator<float> iuvx(*point_msg, "uv_x");
                    sensor_msgs::PointCloud2ConstIterator<float> iuvy(*point_msg, "uv_y");
                    sensor_msgs::PointCloud2ConstIterator<float> iid(*point_msg, "feature_id");
                    const size_t n_pts = static_cast<size_t>(point_msg->width) *
                                         static_cast<size_t>(point_msg->height);
                    point_3d.reserve(n_pts);
                    point_2d_uv.reserve(n_pts);
                    point_2d_normal.reserve(n_pts);
                    point_id.reserve(n_pts);
                    for (size_t i = 0; i < n_pts; ++i,
                                ++ix, ++iy, ++iz,
                                ++inx, ++iny, ++iuvx, ++iuvy, ++iid)
                    {
                        point_3d.push_back(cv::Point3f{*ix, *iy, *iz});
                        point_2d_normal.push_back(cv::Point2f{*inx, *iny});
                        point_2d_uv.push_back(cv::Point2f{*iuvx, *iuvy});
                        point_id.push_back(static_cast<double>(*iid));
                    }

                    const double t_stamp =
                        pose_msg->header.stamp.sec + pose_msg->header.stamp.nanosec * 1e-9;
                    KeyFrame *keyframe = new KeyFrame(
                        t_stamp, frame_index, T, R, image,
                        point_3d, point_2d_uv, point_2d_normal, point_id, sequence);
                    if (USE_DL_LOOP)
                    {
                        computeDinoGlobalDesc(keyframe, image);
                        std::lock_guard<std::mutex> lk(m_desc_);
                        keyframe->attachDLFeatures(id_to_descriptor_);
                    }
                    {
                        std::lock_guard<std::mutex> lk(m_process);
                        start_flag = true;
                        posegraph.addKeyFrame(keyframe, true);
                    }
                    frame_index++;
                    last_t = T;
                }
            }
            std::this_thread::sleep_for(5ms);
        }
    }

    void LoopFusionComponent::onSaveRequest(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        {
            std::lock_guard<std::mutex> lk(m_process);
            posegraph.savePoseGraph();
        }
        res->success = true;
        res->message = "pose graph saved to " + POSE_GRAPH_SAVE_PATH;
        RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
    }

    void LoopFusionComponent::onNewSequenceRequest(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        newSequence();
        res->success = true;
        res->message = "new sequence started";
    }

} // namespace uosm::perception

RCLCPP_COMPONENTS_REGISTER_NODE(uosm::perception::LoopFusionComponent)
