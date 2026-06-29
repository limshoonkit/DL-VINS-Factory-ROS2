#ifndef METRICS_LOGGER_HPP_
#define METRICS_LOGGER_HPP_

#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
#include <mutex>
#include <memory>
#include <filesystem>

namespace uosm
{
    namespace utility
    {
        /**
         * @brief Global parameters for metrics logging shared across components
         */
        struct MetricsConfig
        {
            bool enable_csv_logging = false;
            std::string log_folder = "./tmp/dl_vins_logs";
            std::string session_id; // Unique identifier for this run

            MetricsConfig()
            {
                // Generate session ID from timestamp
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
                session_id = ss.str();
            }
        };

        /**
         * @brief Estimator metrics per optimized frame
         */
        struct EstimatorMetrics
        {
            int frame_id = 0;
            double timestamp = 0.0;

            // Feature health
            int tracked_features = 0;
            int mature_features = 0;          // obs >= 4 (depth solved)
            int triangulated_features = 0;    // estimated_depth > 0
            int stereo_anchored_features = 0; // any obs with is_stereo == true
            double parallax = 0.0;

            // Solver health
            int solver_iterations = 0;
            double solver_time_ms = 0.0;
            double backend_total_time_ms = 0.0;
            double final_cost = 0.0;
            int converged = 0;

            // Cost contribution by factor group (attributes an abnormal final_cost).
            double cost_marg = 0.0;
            double cost_imu = 0.0;
            double cost_temporal = 0.0;  // left cam, two frames (ProjectionTwoFrameOneCamFactor)
            double cost_stereo = 0.0;    // left/right pair, one frame (ProjectionOneFrameTwoCamFactor)
            double cost_cross_cam = 0.0; // left@i vs right@j (ProjectionTwoFrameTwoCamFactor)

            // State norms (drift / IMU bias indicators)
            double velocity_norm = 0.0;
            double accel_bias_norm = 0.0;
            double gyro_bias_norm = 0.0;

            // Timings
            double estimator_total_time_ms = 0.0;

            // Failure detector reason (0 = none; see Estimator::FailureReason)
            int failure_reason = 0;
        };

        /**
         * @brief Feature Tracker metrics per frame
         */
        struct FeatureTrackerMetrics
        {
            int frame_id = 0;
            double timestamp = 0.0;

            // Temporal tracking
            int forward_success = 0;          // left-cam features that survived temporal tracking
            int forward_total = 0;            // denominator for tracking continuity
            int ransac_rejected_temporal = 0; // prev-left -> cur-left E-matrix RANSAC outliers

            // Stereo matching
            int stereo_candidates = 0;
            int stereo_success = 0;
            int stereo_gate_depth_rejects = 0;
            int stereo_gate_reproj_rejects = 0;

            // Feature counts
            int new_features = 0;
            int total_features = 0;
            int mask_culled = 0;

            // Timings
            double extraction_time_ms = 0.0;
            double matcher_time_ms = 0.0;
            double total_time_ms = 0.0;
        };

        class MetricsLogger
        {
        public:
            static MetricsLogger &getInstance()
            {
                static MetricsLogger instance;
                return instance;
            }

            void configure(const MetricsConfig &config, const std::string & = "")
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

                    // Create log folder if it doesn't exist
                    std::filesystem::create_directories(config_.log_folder);
                    std::cout << "[MetricsLogger] Logging enabled, folder: " << config_.log_folder
                              << ", session: " << config_.session_id << std::endl;
                }
            }

            bool isEnabled() const
            {
                return config_.enable_csv_logging;
            }

            // Feature Tracker logging
            void initFeatureTrackerLog(const std::string &method_name)
            {
                if (!config_.enable_csv_logging)
                    return;

                std::lock_guard<std::mutex> lock(mutex_);

                std::string csv_path = config_.log_folder + "/feature_tracker_" +
                                       method_name + "_" + config_.session_id + ".csv";
                feature_tracker_file_.open(csv_path);

                if (feature_tracker_file_.is_open())
                {
                    feature_tracker_file_ << "frame_id,timestamp,"
                                          << "forward_success,forward_total,"
                                          << "ransac_rejected_temporal,"
                                          << "stereo_candidates,stereo_success,"
                                          << "stereo_gate_depth_rejects,stereo_gate_reproj_rejects,"
                                          << "new_features,total_features,"
                                          << "extraction_time_ms,matcher_time_ms,total_time_ms\n";
                    std::cout << "[MetricsLogger] Feature tracker CSV: " << csv_path << std::endl;
                }
                else
                {
                    std::cerr << "[MetricsLogger] Failed to open: " << csv_path << std::endl;
                }
            }

            void logFeatureTracker(const FeatureTrackerMetrics &m)
            {
                if (!config_.enable_csv_logging || !feature_tracker_file_.is_open())
                    return;

                std::lock_guard<std::mutex> lock(mutex_);

                feature_tracker_file_ << m.frame_id << ","
                                      << std::fixed << std::setprecision(6) << m.timestamp << ","
                                      << m.forward_success << ","
                                      << m.forward_total << ","
                                      << m.ransac_rejected_temporal << ","
                                      << m.stereo_candidates << ","
                                      << m.stereo_success << ","
                                      << m.stereo_gate_depth_rejects << ","
                                      << m.stereo_gate_reproj_rejects << ","
                                      << m.new_features << ","
                                      << m.total_features << ","
                                      << m.extraction_time_ms << ","
                                      << m.matcher_time_ms << ","
                                      << m.total_time_ms << "\n";
                feature_tracker_file_.flush();
            }

            // Estimator logging
            void initEstimatorLog(const std::string &tag = "")
            {
                if (!config_.enable_csv_logging)
                    return;

                std::lock_guard<std::mutex> lock(mutex_);
                std::string suffix = tag.empty() ? "" : ("_" + tag);
                std::string csv_path = config_.log_folder + "/estimator_metrics" +
                                       suffix + "_" + config_.session_id + ".csv";
                estimator_file_.open(csv_path);

                if (estimator_file_.is_open())
                {
                    estimator_file_ << "frame_id,timestamp,"
                                    << "tracked_features,mature_features,triangulated_features,"
                                    << "stereo_anchored_features,parallax,"
                                    << "solver_iterations,solver_time_ms,backend_total_time_ms,"
                                    << "final_cost,converged,"
                                    << "cost_marg,cost_imu,"
                                    << "cost_temporal,cost_stereo,cost_cross_cam,"
                                    << "velocity_norm,accel_bias_norm,gyro_bias_norm,"
                                    << "estimator_total_time_ms,failure_reason\n";
                    std::cout << "[MetricsLogger] Estimator CSV: " << csv_path << std::endl;
                }
                else
                {
                    std::cerr << "[MetricsLogger] Failed to open: " << csv_path << std::endl;
                }
            }

            void logEstimator(const EstimatorMetrics &m)
            {
                if (!config_.enable_csv_logging || !estimator_file_.is_open())
                    return;

                std::lock_guard<std::mutex> lock(mutex_);

                estimator_file_ << m.frame_id << ","
                                << std::fixed << std::setprecision(6) << m.timestamp << ","
                                << m.tracked_features << ","
                                << m.mature_features << "," << m.triangulated_features << ","
                                << m.stereo_anchored_features << ","
                                << std::setprecision(6) << m.parallax << ","
                                << m.solver_iterations << ","
                                << std::setprecision(3) << m.solver_time_ms << ","
                                << m.backend_total_time_ms << ","
                                << std::setprecision(6) << m.final_cost << ","
                                << m.converged << ","
                                << m.cost_marg << "," << m.cost_imu << ","
                                << m.cost_temporal << "," << m.cost_stereo << "," << m.cost_cross_cam << ","
                                << m.velocity_norm << "," << m.accel_bias_norm << "," << m.gyro_bias_norm << ","
                                << std::setprecision(3) << m.estimator_total_time_ms << ","
                                << m.failure_reason << "\n";
                estimator_file_.flush();
            }

            void shutdown()
            {
                std::lock_guard<std::mutex> lock(mutex_);

                if (feature_tracker_file_.is_open())
                {
                    feature_tracker_file_.close();
                    std::cout << "[MetricsLogger] Feature tracker log closed" << std::endl;
                }
                if (estimator_file_.is_open())
                {
                    estimator_file_.close();
                    std::cout << "[MetricsLogger] Estimator log closed" << std::endl;
                }
            }

            ~MetricsLogger()
            {
                shutdown();
            }

        private:
            MetricsLogger() = default;
            MetricsLogger(const MetricsLogger &) = delete;
            MetricsLogger &operator=(const MetricsLogger &) = delete;

            MetricsConfig config_;
            std::ofstream feature_tracker_file_;
            std::ofstream estimator_file_;
            std::mutex mutex_;
        };

    } // namespace utility
} // namespace uosm

#endif // METRICS_LOGGER_HPP_
