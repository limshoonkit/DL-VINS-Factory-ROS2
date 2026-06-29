#ifndef LOOP_FUSION_LOOP_METRICS_LOGGER_HPP_
#define LOOP_FUSION_LOOP_METRICS_LOGGER_HPP_

#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <filesystem>

namespace uosm
{
    namespace loop_fusion
    {
        struct LoopFusionConfig
        {
            bool enable_csv_logging = false;
            std::string log_folder = "./tmp/dl_vins_logs";
            std::string session_id; // populated from timestamp at construction

            LoopFusionConfig()
            {
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
                session_id = ss.str();
            }
        };

        struct LoopFusionMetrics
        {
            double timestamp = 0.0;   // keyframe time_stamp
            int kf_index = 0;         // KeyFrame::index assigned by PoseGraph
            int candidate_index = -1; // matched/closed loop frame index (-1 if none)
            int closed = 0;           // findConnection() passed all gates
            int num_inliers = -1;     // PnP RANSAC inliers (-1 if not attempted)
            double top1_sim = -1.0;   // best cosine sim to any eligible prior KF;
                                      // threshold-independent (-1 outside global_vpr)
            double vpr_ms = -1.0;     // DINO+VLAD inference time for this keyframe
            double proc_ms = -1.0;    // per-frame loop processing (detect + connect)
        };

        // Per-optimization record. One row each time the pose graph is solved.
        struct LoopOptimizationMetrics
        {
            int kf_index = 0;
            double timestamp = 0.0;
            double optimize_ms = 0.0; // pose-graph optimization (PGO) solve time
            double drift_xy_m = 0.0;
            double drift_yaw_deg = 0.0;
            int num_loops_optimized = 0; // keyframes carrying has_loop in this opt
        };

        class LoopMetricsLogger
        {
        public:
            static LoopMetricsLogger &getInstance()
            {
                static LoopMetricsLogger instance;
                return instance;
            }

            void configure(const LoopFusionConfig &config)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                config_ = config;

                if (config_.enable_csv_logging)
                {
                    std::filesystem::path log_path(config_.log_folder);
                    if (log_path.is_relative())
                    {
                        log_path = std::filesystem::current_path() / log_path;
                    }
                    config_.log_folder = log_path.string();
                    std::filesystem::create_directories(config_.log_folder);
                    std::cout << "[LoopMetricsLogger] Logging enabled, folder: " << config_.log_folder
                              << ", session: " << config_.session_id << std::endl;
                }
            }

            void setEnabled(bool enabled)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                config_.enable_csv_logging = enabled;
            }

            bool isEnabled() const { return config_.enable_csv_logging; }

            const LoopFusionConfig &getConfig() const { return config_; }

            void initLoopFusionLog(const std::string &tag = "")
            {
                if (!config_.enable_csv_logging) return;
                std::lock_guard<std::mutex> lock(mutex_);
                const std::string suffix = tag.empty() ? "" : ("_" + tag);
                const std::string csv_path = config_.log_folder + "/loop_fusion_metrics" +
                                             suffix + "_" + config_.session_id + ".csv";
                loop_fusion_file_.open(csv_path);
                if (loop_fusion_file_.is_open())
                {
                    loop_fusion_file_ << "timestamp,kf_index,candidate_index,closed,"
                                      << "num_inliers,top1_sim,vpr_ms,proc_ms\n";
                    loop_fusion_file_.flush();
                    std::cout << "[LoopMetricsLogger] loop_fusion CSV: " << csv_path << std::endl;
                }
                else
                {
                    std::cerr << "[LoopMetricsLogger] Failed to open: " << csv_path << std::endl;
                }
            }

            void logLoopFusion(const LoopFusionMetrics &m)
            {
                if (!config_.enable_csv_logging || !loop_fusion_file_.is_open()) return;
                std::lock_guard<std::mutex> lock(mutex_);
                loop_fusion_file_ << std::fixed << std::setprecision(6) << m.timestamp << ","
                                  << m.kf_index << "," << m.candidate_index << ","
                                  << m.closed << "," << m.num_inliers << ","
                                  << std::setprecision(4) << m.top1_sim << ","
                                  << std::setprecision(3) << m.vpr_ms << ","
                                  << m.proc_ms << "\n";
                loop_fusion_file_.flush();
            }

            void initLoopOptimizationLog(const std::string &tag = "")
            {
                if (!config_.enable_csv_logging) return;
                std::lock_guard<std::mutex> lock(mutex_);
                const std::string suffix = tag.empty() ? "" : ("_" + tag);
                const std::string csv_path = config_.log_folder + "/loop_optimizations" +
                                             suffix + "_" + config_.session_id + ".csv";
                loop_optimization_file_.open(csv_path);
                if (loop_optimization_file_.is_open())
                {
                    loop_optimization_file_ << "kf_index,timestamp,optimize_ms,"
                                            << "drift_xy_m,drift_yaw_deg,num_loops_optimized\n";
                    loop_optimization_file_.flush();
                    std::cout << "[LoopMetricsLogger] loop_optimizations CSV: " << csv_path << std::endl;
                }
                else
                {
                    std::cerr << "[LoopMetricsLogger] Failed to open: " << csv_path << std::endl;
                }
            }

            void logLoopOptimization(const LoopOptimizationMetrics &m)
            {
                if (!config_.enable_csv_logging || !loop_optimization_file_.is_open()) return;
                std::lock_guard<std::mutex> lock(mutex_);
                loop_optimization_file_ << m.kf_index << ","
                                        << std::fixed << std::setprecision(6) << m.timestamp << ","
                                        << std::setprecision(3) << m.optimize_ms << ","
                                        << std::setprecision(6) << m.drift_xy_m << ","
                                        << m.drift_yaw_deg << ","
                                        << m.num_loops_optimized << "\n";
                loop_optimization_file_.flush();
            }

            void shutdown()
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (loop_fusion_file_.is_open())
                {
                    loop_fusion_file_.close();
                    std::cout << "[LoopMetricsLogger] loop_fusion log closed" << std::endl;
                }
                if (loop_optimization_file_.is_open())
                {
                    loop_optimization_file_.close();
                    std::cout << "[LoopMetricsLogger] loop_optimizations log closed" << std::endl;
                }
            }

            ~LoopMetricsLogger() { shutdown(); }

        private:
            LoopMetricsLogger() = default;
            LoopMetricsLogger(const LoopMetricsLogger &) = delete;
            LoopMetricsLogger &operator=(const LoopMetricsLogger &) = delete;

            LoopFusionConfig config_;
            std::ofstream loop_fusion_file_;
            std::ofstream loop_optimization_file_;
            std::mutex mutex_;
        };

    } // namespace loop_fusion
} // namespace uosm

#endif // LOOP_FUSION_LOOP_METRICS_LOGGER_HPP_
