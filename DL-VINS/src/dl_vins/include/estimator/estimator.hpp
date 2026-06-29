/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef ESTIMATOR_HPP_
#define ESTIMATOR_HPP_

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <stop_token>
#include <thread>
#include <tuple>
#include <vector>
#include <string>
#include <cstdint>

#include "../backend/backend_interface.hpp"
#include "../feature/feature_manager.hpp"
#include "../feature/feature_struct.hpp"
#include "../imu/integration_base.hpp"
#include "../initial/initial_alignment.hpp"
#include "../initial/initial_sfm.hpp"
#include "../initial/motion_estimator.hpp"
#include "../utility/math_util.hpp"

namespace uosm
{
    namespace perception
    {
        /**
         * @brief Configuration for the Estimator
         */
        struct EstimatorConfig
        {
            // Window and timing
            double td = 0.0;          // Camera-IMU time offset (initial value)
            bool estimate_td = false; // Optimize td online in the BA.

            // Gravity
            double gravity_magnitude = 9.81f;

            // IMU noise parameters
            double acc_n = 0.1f;    // Accelerometer noise density
            double gyr_n = 0.01f;   // Gyroscope noise density
            double acc_w = 0.001f;  // Accelerometer random walk
            double gyr_w = 0.0001f; // Gyroscope random walk

            double focal_length = REF_FOCAL_PX;                       // Real camera focal (px).
            double keyframe_min_parallax_ratio = 10.0 / REF_FOCAL_PX; // Keyframe-selection parallax gate, as a normalized ratio (≈ baseline/depth);
            double max_reprojection_error = 3.0f;                     // outliersRejection() drop threshold

            double max_solver_time = 0.04f; // 40ms

            // Camera mode
            bool use_stereo = true; // false = mono + IMU initialization

            // Hard gates (set > 0 to enable; 0 disables):
            double init_max_g_error = 0.5f;    // Reject if ||g| - gravity_magnitude| > this.
            double init_max_abias_norm = 0.0f; // Reject if post-init ||Bas[i]|| exceeds this for any i.
            double init_min_scale = 0.1f;      // Reject if VI-alignment scale < this.
            double init_max_scale = 10.0f;     // Reject if VI-alignment scale > this.
            double init_min_parallax_ratio = 30.0 / REF_FOCAL_PX;
            bool init_use_static_bootstrap = true;
            double init_static_var_threshold = 0.05f;
            double init_static_window_sec = 1.0f;

            // Backend
            int max_iterations = 8U;
            int num_threads = 0; // 0 = auto-detect
            bool estimate_extrinsic = false;
            bool use_gpu = false;
            bool failure_recovery = false;

            // Computed noise squared values
            double acc_n_sq() const { return acc_n * acc_n; }
            double gyr_n_sq() const { return gyr_n * gyr_n; }
            double acc_w_sq() const { return acc_w * acc_w; }
            double gyr_w_sq() const { return gyr_w * gyr_w; }
        };

        /**
         * @brief Main class for the estimator
         */
        class Estimator
        {
        public:
            enum SolverFlag
            {
                INITIAL,
                NON_LINEAR
            };

            enum MarginalizationFlag
            {
                MARGIN_OLD = 0,
                MARGIN_SECOND_NEW = 1
            };

            /**
             * @brief Construct a new Estimator
             * @param config Configuration parameters
             */
            explicit Estimator(const EstimatorConfig &config);

            /**
             * @brief Destructor - ensures clean thread shutdown
             */
            ~Estimator();

            // Disable copy
            Estimator(const Estimator &) = delete;
            Estimator &operator=(const Estimator &) = delete;

            /**
             * @brief Set camera-IMU extrinsic calibration
             * @param ric Rotation from camera to IMU frame [2 cameras]
             * @param tic Translation from camera to IMU frame [2 cameras]
             */
            void setExtrinsic(const std::array<Eigen::Matrix3d, 2> &ric,
                              const std::array<Eigen::Vector3d, 2> &tic);

            /**
             * @brief Start the processing thread
             * Must be called after construction and setExtrinsic
             */
            void startProcessThread();

            /**
             * @brief Stop the processing thread
             */
            void stopProcessThread();

            /**
             * @brief Input IMU measurement (thread-safe)
             * Called from IMU callback at high rate (~200-400Hz)
             * @param t Timestamp
             * @param acc Linear acceleration
             * @param gyr Angular velocity
             */
            void inputIMU(double t, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

            /**
             * @brief Input tracked features (thread-safe)
             * Called from image callback after feature tracking
             * @param t Timestamp
             * @param observations Feature observations from tracker
             */
            void inputFeatures(double t, const ObservationsMap &observations);

            /**
             * @brief Check if estimator has been initialized
             */
            bool isInitialized() const;

            /**
             * @brief Get the latest optimized state
             * @param P Position
             * @param Q Orientation
             * @param V Velocity
             * @param t Timestamp
             */
            void getLatestState(Eigen::Vector3d &P, Eigen::Quaterniond &Q,
                                Eigen::Vector3d &V, double &t) const;

            /**
             * @brief Monotonic sequence number incremented after each marginalization (keyframe) step
             */
            uint64_t getKeyframeSeq() const { return keyframe_seq_.load(std::memory_order_acquire); }

            /**
             * @brief Monotonic sequence number incremented after every backend
             * update
             */
            uint64_t getBackendUpdateSeq() const { return backend_update_seq_.load(std::memory_order_acquire); }

            /**
             * @brief Get the IMU-propagated state (high-rate)
             * @param P Position
             * @param Q Orientation
             * @param V Velocity
             */
            void getIMUPropagatedState(Eigen::Vector3d &P, Eigen::Quaterniond &Q,
                                       Eigen::Vector3d &V) const;

            /**
             * @brief Get active sliding-window landmarks in world frame.
             * Filters tracks with >=2 observations, has_valid_depth, and a
             * start_frame in the mid-window range.
             */
            void getActiveLandmarks(std::vector<Eigen::Vector3d> &out_world) const;

            void getKeyframePose(int idx, Eigen::Vector3d &P, Eigen::Matrix3d &R) const;
            double getHeaderAt(int idx) const;
            void getExtrinsic(int cam_idx, Eigen::Vector3d &tic_out, Eigen::Matrix3d &ric_out) const;
            static constexpr int keyframePublishIndex() { return WINDOW_SIZE - 2; }

            struct KeyframePoint
            {
                Eigen::Vector3d world;
                Eigen::Vector2d norm;
                Eigen::Vector2d uv;
                int feature_id;
            };

            void getKeyframeLandmarks(int kf_idx, std::vector<KeyframePoint> &out) const;
            void getMarginalizedLandmarks(std::vector<Eigen::Vector3d> &out_world) const;

        private:
            /**
             * @brief Main processing loop (runs in separate jthread)
             */
            void processMeasurements(std::stop_token st);

            /**
             * @brief Process IMU measurement for pre-integration and state propagation
             */
            void processIMU(double t, double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

            /**
             * @brief Process image features - main VIO logic
             */
            void processImage(double timestamp, const ObservationsMap &observations);

            /**
             * @brief Get IMU measurements between two timestamps
             */
            bool getIMUInterval(double t0, double t1,
                                std::vector<std::pair<double, Eigen::Vector3d>> &accVector,
                                std::vector<std::pair<double, Eigen::Vector3d>> &gyrVector);

            /**
             * @brief Initialize first pose from IMU (gravity alignment)
             */
            void initFirstIMUPose(std::vector<std::pair<double, Eigen::Vector3d>> &accVector);

            /**
             * @brief Mono + IMU initialization: SFM → visual-inertial alignment
             */
            bool initialStructure();

            /**
             * @brief Apply scale, velocity, and gravity alignment after SFM
             */
            bool visualInitialAlign();

            /**
             * @brief Find relative pose between a previous frame and the newest frame
             * @param relative_R [out] Relative rotation
             * @param relative_T [out] Relative translation
             * @param l          [out] Index of the matched previous frame
             * @return true if sufficient parallax and inliers found
             */
            bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);

            /**
             * @brief Fast IMU propagation for high-rate output
             * Called with mPropagate_ held
             */
            void fastPredictIMU(double t, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

            void slideWindow();
            void slideWindowOld();
            void slideWindowNew();

            void clearState();

            enum class FailureReason : uint8_t
            {
                None = 0,
                NaN = 1u << 0,          // NaN/Inf in pose/velocity/bias -> hard reset
                Velocity = 1u << 1,     // |V| above sanity bound
                PositionJump = 1u << 2, // window-tail position jumped too far in one step
                AccelBias = 1u << 3,    // |Ba| blew up
                GyroBias = 1u << 4,     // |Bg| blew up
            };
            struct FailureReport
            {
                uint8_t mask = 0;   // OR of FailureReason bits (0 == healthy)
                std::string detail; // "value/threshold" for each checked quantity
            };
            FailureReport failureDetection();
            static std::string failureReasonName(uint8_t mask);

            void updateLatestStates();

            void outliersRejection(std::set<int> &removeIndex);
            double reprojectionError(const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Pi,
                                     const Eigen::Matrix3d &rici, const Eigen::Vector3d &tici,
                                     const Eigen::Matrix3d &Rj, const Eigen::Vector3d &Pj,
                                     const Eigen::Matrix3d &ricj, const Eigen::Vector3d &ticj,
                                     double depth, const Eigen::Vector3d &uvi, const Eigen::Vector3d &uvj);

            mutable std::mutex mBuf_;       // Protects acc_buf_, gyr_buf_, feature_buf_
            mutable std::mutex mProcess_;   // Protects state during optimization
            mutable std::mutex mPropagate_; // Protects latest_* for IMU propagation

            std::condition_variable_any buf_cv_; // Wakes process thread on new data
            std::jthread process_thread_;
            std::atomic<bool> initialized_{false};

            std::queue<std::pair<double, Eigen::Vector3d>> acc_buf_, gyr_buf_;
            std::queue<std::pair<double, ObservationsMap>> feature_buf_;

            std::array<Eigen::Vector3d, WINDOW_SIZE + 1> Ps_, Vs_, Bas_, Bgs_;
            std::array<Eigen::Matrix3d, WINDOW_SIZE + 1> Rs_;
            std::array<double, WINDOW_SIZE + 1> Headers_;

            // IMU data buffers for each frame (for re-marginalization)
            std::array<std::vector<double>, WINDOW_SIZE + 1> dt_buf_;
            std::array<std::vector<Eigen::Vector3d>, WINDOW_SIZE + 1> linear_acceleration_buf_;
            std::array<std::vector<Eigen::Vector3d>, WINDOW_SIZE + 1> angular_velocity_buf_;
            std::array<std::unique_ptr<IntegrationBase>, WINDOW_SIZE + 1> pre_integrations_;
            std::shared_ptr<IntegrationBase> tmp_pre_integration_;

            // Current IMU measurement
            Eigen::Vector3d acc_0_, gyr_0_;

            std::shared_ptr<FeatureManager> feature_manager_;
            std::unique_ptr<IBackendSolver> backend_;
            MotionEstimator motion_estimator_;

            std::array<Eigen::Matrix3d, 2> ric_;
            std::array<Eigen::Vector3d, 2> tic_;

            SolverFlag solver_flag_ = INITIAL;
            MarginalizationFlag marginalization_flag_;
            int frame_count_ = 0;
            double prev_time_ = -1.0;
            double cur_time_ = 0.0;
            bool first_imu_ = false;
            bool init_first_pose_flag_ = false;

            // Initial pose (for relative output)
            Eigen::Vector3d init_P_;
            Eigen::Matrix3d init_R_;

            // failureDetection():
            Eigen::Vector3d failure_last_P_{Eigen::Vector3d::Zero()};
            bool failure_last_P_valid_{false};

            // For gyro bias initialization
            std::map<double, ImageFrame> all_image_frame_;

            double latest_time_ = 0.0;
            Eigen::Vector3d latest_P_{Eigen::Vector3d::Zero()};
            Eigen::Vector3d latest_V_{Eigen::Vector3d::Zero()};
            Eigen::Vector3d latest_Ba_{Eigen::Vector3d::Zero()};
            Eigen::Vector3d latest_Bg_{Eigen::Vector3d::Zero()};
            Eigen::Vector3d latest_acc_0_{Eigen::Vector3d::Zero()};
            Eigen::Vector3d latest_gyr_0_{Eigen::Vector3d::Zero()};
            Eigen::Quaterniond latest_Q_{Eigen::Quaterniond::Identity()};
            std::atomic<uint64_t> keyframe_seq_{0};
            std::atomic<uint64_t> backend_update_seq_{0};

            // Static-period bootstrap state. Updated by inputIMU() while solver_flag_ == INITIAL;
            // consumed by initialStructure() to seed Bgs before alignment.
            std::deque<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>> bootstrap_imu_window_;
            bool bootstrap_static_seen_ = false;
            Eigen::Vector3d bootstrap_gyro_mean_ = Eigen::Vector3d::Zero();
            Eigen::Vector3d bootstrap_acc_mean_ = Eigen::Vector3d::Zero();

            // Init-quality metrics
            double init_g_norm_ = 0.0;
            double init_scale_ = 0.0;
            double init_abias_norm_ = 0.0;

            EstimatorConfig config_;
            Eigen::Vector3d gravity_;
        };

    } // namespace perception
} // namespace uosm

#endif // ESTIMATOR_HPP_
