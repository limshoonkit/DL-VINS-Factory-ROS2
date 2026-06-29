/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "../../include/estimator/estimator.hpp"
#include "../../include/utility/metrics_logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <rclcpp/logging.hpp>
#include <rclcpp/clock.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace uosm
{
    namespace perception
    {

        Estimator::Estimator(const EstimatorConfig &config)
            : config_(config),
              gravity_(0, 0, config.gravity_magnitude)
        {
            // Initialize state arrays
            for (int i = 0; i < WINDOW_SIZE + 1; i++)
            {
                Ps_[i].setZero();
                Vs_[i].setZero();
                Rs_[i].setIdentity();
                Bas_[i].setZero();
                Bgs_[i].setZero();
                Headers_[i] = 0.0;
            }

            // Initialize extrinsics to identity
            for (int i = 0; i < 2; i++)
            {
                ric_[i].setIdentity();
                tic_[i].setZero();
            }

            // Initialize IMU measurements
            acc_0_.setZero();
            gyr_0_.setZero();
            init_P_.setZero();
            init_R_.setIdentity();

            // Create feature manager
            feature_manager_ = std::make_shared<FeatureManager>(
                config_.focal_length,
                config_.keyframe_min_parallax_ratio);

            // Create backend solver
            BackendSolverConfig backend_config;
            backend_config.focal_length = config_.focal_length;
            backend_config.max_iterations = config_.max_iterations;
            backend_config.max_solver_time = config_.max_solver_time;
            backend_config.estimate_extrinsic = config_.estimate_extrinsic;
            backend_config.initial_td = config_.td;
            backend_config.estimate_td = config_.estimate_td;
            backend_config.use_gpu = config_.use_gpu;
            backend_config.num_threads = config_.num_threads;

            backend_ = BackendSolverFactory::create(BackendType::CERES, backend_config);

            motion_estimator_.setFocalLength(config_.focal_length);
        }

        Estimator::~Estimator()
        {
            stopProcessThread();
        }

        void Estimator::setExtrinsic(const std::array<Eigen::Matrix3d, 2> &ric,
                                     const std::array<Eigen::Vector3d, 2> &tic)
        {
            ric_ = ric;
            tic_ = tic;

            // Pass extrinsics to feature manager
            feature_manager_->setRic(ric_.data());
        }

        void Estimator::startProcessThread()
        {
            if (process_thread_.joinable())
                return;

            auto &logger = uosm::utility::MetricsLogger::getInstance();
            if (logger.isEnabled())
            {
                logger.initEstimatorLog();
            }

            process_thread_ = std::jthread([this](std::stop_token st)
                                           { processMeasurements(st); });
            RCLCPP_INFO(rclcpp::get_logger("estimator"), "Processing thread started");
        }

        void Estimator::stopProcessThread()
        {
            if (process_thread_.joinable())
            {
                process_thread_.request_stop();
                buf_cv_.notify_all();
                process_thread_.join();
                RCLCPP_INFO(rclcpp::get_logger("estimator"), "Processing thread stopped");
            }
        }

        void Estimator::inputIMU(double t, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
        {
            {
                std::lock_guard lock(mBuf_);
                acc_buf_.push({t, acc});
                gyr_buf_.push({t, gyr});

                if (config_.init_use_static_bootstrap && !bootstrap_static_seen_ &&
                    solver_flag_ == INITIAL)
                {
                    bootstrap_imu_window_.emplace_back(t, acc, gyr);
                    while (!bootstrap_imu_window_.empty() &&
                           t - std::get<0>(bootstrap_imu_window_.front()) >
                               config_.init_static_window_sec * 2.0)
                    {
                        bootstrap_imu_window_.pop_front();
                    }

                    if (!bootstrap_imu_window_.empty() &&
                        t - std::get<0>(bootstrap_imu_window_.front()) >=
                            config_.init_static_window_sec)
                    {
                        Eigen::Vector3d acc_sum = Eigen::Vector3d::Zero();
                        Eigen::Vector3d gyr_sum = Eigen::Vector3d::Zero();
                        for (const auto &s : bootstrap_imu_window_)
                        {
                            acc_sum += std::get<1>(s);
                            gyr_sum += std::get<2>(s);
                        }
                        const double n = static_cast<double>(bootstrap_imu_window_.size());
                        const Eigen::Vector3d acc_mean = acc_sum / n;
                        const Eigen::Vector3d gyr_mean = gyr_sum / n;

                        double acc_var = 0.0;
                        for (const auto &s : bootstrap_imu_window_)
                            acc_var += (std::get<1>(s) - acc_mean).squaredNorm();
                        acc_var = std::sqrt(acc_var / n);

                        if (acc_var < config_.init_static_var_threshold)
                        {
                            bootstrap_static_seen_ = true;
                            bootstrap_gyro_mean_ = gyr_mean;
                            bootstrap_acc_mean_ = acc_mean;
                            RCLCPP_INFO(rclcpp::get_logger("estimator"),
                                        "[bootstrap] static window captured (acc_var=%.4f): "
                                        "Bgs_prior=[%.5f, %.5f, %.5f]",
                                        acc_var, gyr_mean.x(), gyr_mean.y(), gyr_mean.z());
                        }
                    }
                }
            }
            buf_cv_.notify_one();

            if (solver_flag_ == NON_LINEAR)
            {
                std::lock_guard lock(mPropagate_);
                fastPredictIMU(t, acc, gyr);
            }
        }

        void Estimator::inputFeatures(double t, const ObservationsMap &observations)
        {
            {
                std::lock_guard lock(mBuf_);
                feature_buf_.push({t, observations});
            }
            buf_cv_.notify_one();
        }

        bool Estimator::isInitialized() const
        {
            return initialized_.load(std::memory_order_acquire);
        }

        void Estimator::getLatestState(Eigen::Vector3d &P, Eigen::Quaterniond &Q,
                                       Eigen::Vector3d &V, double &t) const
        {
            std::lock_guard<std::mutex> lock(mPropagate_);
            P = latest_P_;
            Q = latest_Q_;
            V = latest_V_;
            t = latest_time_;
        }

        void Estimator::getIMUPropagatedState(Eigen::Vector3d &P, Eigen::Quaterniond &Q,
                                              Eigen::Vector3d &V) const
        {
            std::lock_guard<std::mutex> lock(mPropagate_);
            P = latest_P_;
            Q = latest_Q_;
            V = latest_V_;
        }

        void Estimator::getActiveLandmarks(std::vector<Eigen::Vector3d> &out_world) const
        {
            std::lock_guard<std::mutex> lock(mProcess_);
            out_world.clear();
            if (!feature_manager_ || solver_flag_ != NON_LINEAR)
                return;

            const auto &features = feature_manager_->getFeatures();
            out_world.reserve(features.size());
            const int window_size = WINDOW_SIZE;
            for (const auto &[id, track] : features)
            {
                const int used_num = static_cast<int>(track.observations.size());
                if (used_num < 2 || track.start_frame >= window_size - 2)
                    continue;
                if (track.start_frame > window_size * 3 / 4 || !track.has_valid_depth)
                    continue;
                const int imu_i = track.start_frame;
                const Eigen::Vector3d pts_i =
                    track.observations.front().left().point_c * track.estimated_depth;
                const Eigen::Vector3d w_pts =
                    Rs_[imu_i] * (ric_[0] * pts_i + tic_[0]) + Ps_[imu_i];
                out_world.push_back(w_pts);
            }
        }

        void Estimator::getKeyframePose(int idx, Eigen::Vector3d &P, Eigen::Matrix3d &R) const
        {
            std::lock_guard<std::mutex> lock(mProcess_);
            P = Ps_[idx];
            R = Rs_[idx];
        }

        double Estimator::getHeaderAt(int idx) const
        {
            std::lock_guard<std::mutex> lock(mProcess_);
            return Headers_[idx];
        }

        void Estimator::getExtrinsic(int cam_idx, Eigen::Vector3d &tic_out, Eigen::Matrix3d &ric_out) const
        {
            std::lock_guard<std::mutex> lock(mProcess_);
            tic_out = tic_[cam_idx];
            ric_out = ric_[cam_idx];
        }

        void Estimator::getKeyframeLandmarks(int kf_idx, std::vector<KeyframePoint> &out) const
        {
            std::lock_guard<std::mutex> lock(mProcess_);
            out.clear();
            if (!feature_manager_ || solver_flag_ != NON_LINEAR)
                return;

            const auto &features = feature_manager_->getFeatures();
            out.reserve(features.size());
            for (const auto &[id, track] : features)
            {
                if (!track.has_valid_depth)
                    continue;
                const int end_frame = track.start_frame + static_cast<int>(track.observations.size()) - 1;
                if (track.start_frame > kf_idx || end_frame < kf_idx)
                    continue;

                const int imu_i = track.start_frame;
                const Eigen::Vector3d pts_i =
                    track.observations.front().left().point_c * track.estimated_depth;
                KeyframePoint kp;
                kp.world = Rs_[imu_i] * (ric_[0] * pts_i + tic_[0]) + Ps_[imu_i];

                const auto &obs_kf = track.observations[kf_idx - track.start_frame].left();
                kp.norm.x() = obs_kf.point_c.x();
                kp.norm.y() = obs_kf.point_c.y();
                kp.uv = obs_kf.uv;
                kp.feature_id = id;
                out.push_back(kp);
            }
        }

        void Estimator::getMarginalizedLandmarks(std::vector<Eigen::Vector3d> &out_world) const
        {
            std::lock_guard<std::mutex> lock(mProcess_);
            out_world.clear();
            if (!feature_manager_ || solver_flag_ != NON_LINEAR)
                return;

            const auto &features = feature_manager_->getFeatures();
            for (const auto &[id, track] : features)
            {
                if (!track.has_valid_depth || track.start_frame != 0)
                    continue;
                const Eigen::Vector3d pts_i =
                    track.observations.front().left().point_c * track.estimated_depth;
                out_world.push_back(Rs_[0] * (ric_[0] * pts_i + tic_[0]) + Ps_[0]);
            }
        }

        void Estimator::processMeasurements(std::stop_token st)
        {
            while (!st.stop_requested())
            {
                std::pair<double, ObservationsMap> feature;
                std::vector<std::pair<double, Eigen::Vector3d>> accVector, gyrVector;

                // Wait for features to arrive
                {
                    std::unique_lock lock(mBuf_);
                    buf_cv_.wait(lock, st, [&]
                                 { return !feature_buf_.empty(); });
                    if (st.stop_requested())
                        break;
                }

                // Wait for IMU data covering the feature timestamp
                {
                    std::unique_lock lock(mBuf_);
                    double feature_time = feature_buf_.front().first + config_.td;
                    buf_cv_.wait(lock, st, [&]
                                 { return !acc_buf_.empty() && acc_buf_.back().first >= feature_time; });
                    if (st.stop_requested())
                        break;

                    feature = feature_buf_.front();
                    feature_buf_.pop();
                    cur_time_ = feature.first + config_.td;
                    if (!getIMUInterval(prev_time_, cur_time_, accVector, gyrVector) ||
                        accVector.empty() || gyrVector.size() != accVector.size())
                    {
                        RCLCPP_WARN(rclcpp::get_logger("estimator"),
                                    "Skipping frame %.6f: insufficient IMU coverage",
                                    feature.first);
                        continue;
                    }
                }

                if (!init_first_pose_flag_)
                {
                    initFirstIMUPose(accVector);
                }

                for (size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if (i == 0)
                        dt = accVector[i].first - prev_time_;
                    else if (i == accVector.size() - 1)
                        dt = cur_time_ - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;

                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }

                {
                    std::lock_guard lock(mProcess_);
                    processImage(feature.first, feature.second);
                    prev_time_ = cur_time_;
                }
            }
        }

        bool Estimator::getIMUInterval(double t0, double t1,
                                       std::vector<std::pair<double, Eigen::Vector3d>> &accVector,
                                       std::vector<std::pair<double, Eigen::Vector3d>> &gyrVector)
        {
            if (acc_buf_.empty())
            {
                RCLCPP_DEBUG(rclcpp::get_logger("estimator"), "No IMU data received");
                return false;
            }

            if (t1 <= acc_buf_.back().first)
            {
                while (acc_buf_.front().first <= t0)
                {
                    acc_buf_.pop();
                    gyr_buf_.pop();
                }
                while (acc_buf_.front().first < t1)
                {
                    accVector.push_back(acc_buf_.front());
                    acc_buf_.pop();
                    gyrVector.push_back(gyr_buf_.front());
                    gyr_buf_.pop();
                }
                if (acc_buf_.empty())
                    return false;
                accVector.push_back(acc_buf_.front());
                gyrVector.push_back(gyr_buf_.front());
            }
            else
            {
                RCLCPP_DEBUG(rclcpp::get_logger("estimator"), "Waiting for IMU data...");
                return false;
            }
            return true;
        }

        void Estimator::processIMU(double t, double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
        {
            if (!first_imu_)
            {
                first_imu_ = true;
                acc_0_ = acc;
                gyr_0_ = gyr;
            }

            // Create pre-integration if not exists
            if (!pre_integrations_[frame_count_])
            {
                pre_integrations_[frame_count_] = std::make_unique<IntegrationBase>(
                    acc_0_, gyr_0_, Bas_[frame_count_], Bgs_[frame_count_],
                    gravity_, config_.acc_n_sq(), config_.acc_w_sq(),
                    config_.gyr_n_sq(), config_.gyr_w_sq());
            }

            if (frame_count_ != 0)
            {
                pre_integrations_[frame_count_]->push_back(dt, acc, gyr);

                if (tmp_pre_integration_)
                {
                    tmp_pre_integration_->push_back(dt, acc, gyr);
                }

                // Store IMU data for potential re-marginalization
                dt_buf_[frame_count_].push_back(dt);
                linear_acceleration_buf_[frame_count_].push_back(acc);
                angular_velocity_buf_[frame_count_].push_back(gyr);

                // Propagate state using midpoint integration
                int j = frame_count_;
                Eigen::Vector3d un_acc_0 = Rs_[j] * (acc_0_ - Bas_[j]) - gravity_;
                Eigen::Vector3d un_gyr = 0.5 * (gyr_0_ + gyr) - Bgs_[j];
                Rs_[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rs_[j] * (acc - Bas_[j]) - gravity_;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ps_[j] += dt * Vs_[j] + 0.5 * dt * dt * un_acc;
                Vs_[j] += dt * un_acc;
            }

            acc_0_ = acc;
            gyr_0_ = gyr;
        }

        void Estimator::processImage(double timestamp, const ObservationsMap &observations)
        {
            // Add features and decide keyframe vs non-keyframe
            if (feature_manager_->addFeatureCheckParallax(frame_count_, observations, config_.td))
            {
                marginalization_flag_ = MARGIN_OLD;
            }
            else
            {
                marginalization_flag_ = MARGIN_SECOND_NEW;
            }

            Headers_[frame_count_] = timestamp;

            ImageFrame imageframe(timestamp);
            imageframe.pre_integration = tmp_pre_integration_;
            imageframe.R = Rs_[frame_count_];
            imageframe.T = Ps_[frame_count_];
            all_image_frame_[timestamp] = imageframe;

            tmp_pre_integration_ = std::make_shared<IntegrationBase>(
                acc_0_, gyr_0_, Bas_[frame_count_], Bgs_[frame_count_],
                gravity_, config_.acc_n_sq(), config_.acc_w_sq(),
                config_.gyr_n_sq(), config_.gyr_w_sq());

            if (solver_flag_ == INITIAL)
            {
                if (config_.use_stereo)
                {
                    // Initialize frame pose using PnP (for frames with 3D features)
                    feature_manager_->initFramePoseByPnP(frame_count_, Ps_.data(), Rs_.data(),
                                                         tic_.data(), ric_.data());

                    // Triangulate new features using stereo
                    feature_manager_->triangulate(frame_count_, Ps_.data(), Rs_.data(),
                                                  tic_.data(), ric_.data());

                    if (frame_count_ == WINDOW_SIZE)
                    {
                        // Gyro-bias-only initialization
                        int idx = 0;
                        for (auto &kv : all_image_frame_)
                        {
                            kv.second.R = Rs_[idx];
                            kv.second.T = Ps_[idx];
                            idx++;
                        }
                        VisualIMUAlignment::solveGyroscopeBias(
                            all_image_frame_, Bgs_.data(), WINDOW_SIZE);
                        for (int i = 0; i <= WINDOW_SIZE; i++)
                        {
                            if (pre_integrations_[i])
                                pre_integrations_[i]->repropagate(
                                    Eigen::Vector3d::Zero(), Bgs_[i]);
                        }

                        // Run optimization
                        std::vector<std::unique_ptr<IntegrationBase>> pre_int_vec;
                        pre_int_vec.reserve(WINDOW_SIZE + 1);
                        for (int i = 0; i <= WINDOW_SIZE; i++)
                            pre_int_vec.push_back(std::move(pre_integrations_[i]));

                        backend_->solve(frame_count_, Ps_.data(), Rs_.data(), Vs_.data(),
                                        Bas_.data(), Bgs_.data(), tic_.data(), ric_.data(),
                                        pre_int_vec, feature_manager_, marginalization_flag_);
                        backend_->marginalize(frame_count_, Ps_.data(), Rs_.data(), Vs_.data(),
                                              Bas_.data(), Bgs_.data(), tic_.data(), ric_.data(),
                                              pre_int_vec, feature_manager_,
                                              marginalization_flag_ == MARGIN_OLD);

                        for (int i = 0; i <= WINDOW_SIZE; i++)
                            pre_integrations_[i] = std::move(pre_int_vec[i]);

                        updateLatestStates();
                        solver_flag_ = NON_LINEAR;
                        initialized_.store(true, std::memory_order_release);

                        init_P_ = Ps_[0];
                        init_R_ = Rs_[0];

                        RCLCPP_INFO(rclcpp::get_logger("estimator"), "Stereo initialization complete!");
                        slideWindow();
                    }
                }
                else
                {
                    if (frame_count_ == WINDOW_SIZE)
                    {
                        bool result = false;
                        if ((timestamp - Headers_[0]) > 0.1)
                        {
                            result = initialStructure();
                        }
                        if (result)
                        {
                            // Run first optimization
                            std::vector<std::unique_ptr<IntegrationBase>> pre_int_vec;
                            pre_int_vec.reserve(WINDOW_SIZE + 1);
                            for (int i = 0; i <= WINDOW_SIZE; i++)
                                pre_int_vec.push_back(std::move(pre_integrations_[i]));

                            backend_->solve(frame_count_, Ps_.data(), Rs_.data(), Vs_.data(),
                                            Bas_.data(), Bgs_.data(), tic_.data(), ric_.data(),
                                            pre_int_vec, feature_manager_, marginalization_flag_);

                            // Capture post-init accel bias and optionally gate (opt-in).
                            double max_abias = 0.0;
                            for (int i = 0; i <= WINDOW_SIZE; i++)
                                max_abias = std::max(max_abias, Bas_[i].norm());
                            init_abias_norm_ = max_abias;

                            const bool abias_gate_failed =
                                config_.init_max_abias_norm > 0.0 &&
                                max_abias > config_.init_max_abias_norm;

                            const bool scale_gate_failed =
                                (config_.init_min_scale > 0.0 && init_scale_ < config_.init_min_scale) ||
                                (config_.init_max_scale > 0.0 && init_scale_ > config_.init_max_scale);

                            if (abias_gate_failed || scale_gate_failed)
                            {
                                if (scale_gate_failed)
                                    RCLCPP_WARN(rclcpp::get_logger("estimator"),
                                                "Mono init rejected: scale=%.4f outside [%.3f, %.3f]. Clearing state.",
                                                init_scale_, config_.init_min_scale, config_.init_max_scale);
                                else
                                    RCLCPP_WARN(rclcpp::get_logger("estimator"),
                                                "Mono init rejected: |Bas|=%.3f > %.3f. Clearing state.",
                                                max_abias, config_.init_max_abias_norm);
                                // Return pre_integrations so clearState can release them cleanly.
                                for (int i = 0; i <= WINDOW_SIZE; i++)
                                    pre_integrations_[i] = std::move(pre_int_vec[i]);
                                clearState();
                            }
                            else
                            {
                                backend_->marginalize(frame_count_, Ps_.data(), Rs_.data(), Vs_.data(),
                                                      Bas_.data(), Bgs_.data(), tic_.data(), ric_.data(),
                                                      pre_int_vec, feature_manager_,
                                                      marginalization_flag_ == MARGIN_OLD);

                                for (int i = 0; i <= WINDOW_SIZE; i++)
                                    pre_integrations_[i] = std::move(pre_int_vec[i]);

                                updateLatestStates();
                                solver_flag_ = NON_LINEAR;
                                initialized_.store(true, std::memory_order_release);

                                init_P_ = Ps_[0];
                                init_R_ = Rs_[0];

                                RCLCPP_INFO(rclcpp::get_logger("estimator"), "Mono initialization complete!");
                                RCLCPP_INFO(rclcpp::get_logger("estimator"),
                                            "[init_quality] |g|=%.4f scale=%.4f abias=%.3f static_bootstrap=%s",
                                            init_g_norm_, init_scale_, init_abias_norm_,
                                            bootstrap_static_seen_ ? "yes" : "no");
                                slideWindow();
                            }
                        }
                        else
                        {
                            slideWindow();
                        }
                    }
                }

                // Fill window before initialization
                if (frame_count_ < WINDOW_SIZE)
                {
                    frame_count_++;
                    int prev = frame_count_ - 1;
                    Ps_[frame_count_] = Ps_[prev];
                    Vs_[frame_count_] = Vs_[prev];
                    Rs_[frame_count_] = Rs_[prev];
                    Bas_[frame_count_] = Bas_[prev];
                    Bgs_[frame_count_] = Bgs_[prev];
                }
            }
            else
            {
                auto t_processimage_start = std::chrono::high_resolution_clock::now();

                // Triangulate new features
                feature_manager_->triangulate(frame_count_, Ps_.data(), Rs_.data(),
                                              tic_.data(), ric_.data());

                // Convert pre_integrations_ array to vector for backend interface
                std::vector<std::unique_ptr<IntegrationBase>> pre_int_vec;
                pre_int_vec.reserve(WINDOW_SIZE + 1);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_int_vec.push_back(std::move(pre_integrations_[i]));
                }

                // Run optimization
                backend_->solve(frame_count_, Ps_.data(), Rs_.data(), Vs_.data(),
                                Bas_.data(), Bgs_.data(), tic_.data(), ric_.data(),
                                pre_int_vec, feature_manager_, marginalization_flag_);

                // Marginalize
                backend_->marginalize(frame_count_, Ps_.data(), Rs_.data(), Vs_.data(),
                                      Bas_.data(), Bgs_.data(), tic_.data(), ric_.data(),
                                      pre_int_vec, feature_manager_,
                                      marginalization_flag_ == MARGIN_OLD);

                // Move back
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations_[i] = std::move(pre_int_vec[i]);
                }

                // Outlier rejection
                std::set<int> removeIndex;
                outliersRejection(removeIndex);
                feature_manager_->removeOutlier(removeIndex);

                // Failure detection
                const FailureReport fr = failureDetection();
                if (fr.mask != 0)
                {
                    static rclcpp::Clock s_fail_clock(RCL_STEADY_TIME);
                    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("estimator"), s_fail_clock, 1000,
                                         "Failure detected [%s]: %s",
                                         failureReasonName(fr.mask).c_str(), fr.detail.c_str());

                    const bool is_nan = (fr.mask & static_cast<uint8_t>(FailureReason::NaN)) != 0;
                    if (is_nan || config_.failure_recovery)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("estimator"),
                                    "%s -> resetting estimator (re-initialize)",
                                    is_nan ? "Unrecoverable NaN/Inf state"
                                           : "failure_recovery enabled");
                        clearState();
                        return;
                    }
                }

                slideWindow();
                updateLatestStates();

                auto t_processimage_end = std::chrono::high_resolution_clock::now();

                auto &metrics_logger = uosm::utility::MetricsLogger::getInstance();
                if (metrics_logger.isEnabled())
                {
                    static int estimator_frame_counter = 0;
                    auto solver_stats = backend_->getStatistics();

                    int mature = 0, triangulated = 0, stereo_anchored = 0;
                    for (const auto &[fid, track] : feature_manager_->getFeatures())
                    {
                        if (static_cast<int>(track.observations.size()) >= 4)
                            mature++;
                        if (track.estimated_depth > 0)
                            triangulated++;
                        for (const auto &obs : track.observations)
                        {
                            if (obs.is_stereo)
                            {
                                stereo_anchored++;
                                break;
                            }
                        }
                    }

                    uosm::utility::EstimatorMetrics em;
                    em.frame_id = estimator_frame_counter++;
                    em.timestamp = Headers_[frame_count_];
                    em.tracked_features = feature_manager_->lastTrackNum();
                    em.mature_features = mature;
                    em.triangulated_features = triangulated;
                    em.stereo_anchored_features = stereo_anchored;
                    em.parallax = feature_manager_->lastAverageParallax();
                    em.solver_iterations = solver_stats.iterations;
                    em.solver_time_ms = solver_stats.solve_time_ms;
                    em.backend_total_time_ms = solver_stats.solve_time_ms +
                                               solver_stats.marginalization_time_ms;
                    em.final_cost = solver_stats.final_cost;
                    em.converged = solver_stats.converged ? 1 : 0;
                    em.cost_marg = solver_stats.cost_marg;
                    em.cost_imu = solver_stats.cost_imu;
                    em.cost_temporal = solver_stats.cost_temporal;
                    em.cost_stereo = solver_stats.cost_stereo;
                    em.cost_cross_cam = solver_stats.cost_cross_cam;
                    em.velocity_norm = Vs_[frame_count_].norm();
                    em.accel_bias_norm = Bas_[frame_count_].norm();
                    em.gyro_bias_norm = Bgs_[frame_count_].norm();
                    em.estimator_total_time_ms =
                        std::chrono::duration_cast<std::chrono::microseconds>(t_processimage_end - t_processimage_start).count() / 1000.0;
                    em.failure_reason = static_cast<int>(fr.mask);

                    metrics_logger.logEstimator(em);
                }
            }
        }

        void Estimator::initFirstIMUPose(std::vector<std::pair<double, Eigen::Vector3d>> &accVector)
        {
            RCLCPP_INFO(rclcpp::get_logger("estimator"), "Initializing first IMU pose");
            init_first_pose_flag_ = true;

            const int n = static_cast<int>(accVector.size());
            if (n <= 0)
            {
                RCLCPP_WARN(rclcpp::get_logger("estimator"),
                            "initFirstIMUPose: no IMU samples in interval; keeping identity rotation");
                return;
            }

            // Average accelerometer readings to get gravity direction
            Eigen::Vector3d averAcc(0, 0, 0);
            for (int i = 0; i < n; i++)
            {
                averAcc += accVector[i].second;
            }
            averAcc /= static_cast<double>(n);

            RCLCPP_DEBUG(rclcpp::get_logger("estimator"), "Average acc: [%.4f, %.4f, %.4f]", averAcc.x(), averAcc.y(), averAcc.z());

            // Compute rotation to align with gravity
            Eigen::Matrix3d R0 = Utility::g2R(averAcc);
            double yaw = Utility::R2ypr(R0).x();
            R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;

            Rs_[0] = R0;
            RCLCPP_DEBUG(rclcpp::get_logger("estimator"), "Initial R0 set from gravity alignment");
        }

        void Estimator::fastPredictIMU(double t, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
        {
            double dt = t - latest_time_;
            latest_time_ = t;

            Eigen::Vector3d un_acc_0 = latest_Q_ * (latest_acc_0_ - latest_Ba_) - gravity_;
            Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0_ + gyr) - latest_Bg_;
            latest_Q_ = latest_Q_ * Utility::deltaQ(un_gyr * dt);
            latest_Q_.normalize();
            Eigen::Vector3d un_acc_1 = latest_Q_ * (acc - latest_Ba_) - gravity_;
            Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            latest_P_ = latest_P_ + dt * latest_V_ + 0.5 * dt * dt * un_acc;
            latest_V_ = latest_V_ + dt * un_acc;
            latest_acc_0_ = acc;
            latest_gyr_0_ = gyr;
        }

        void Estimator::updateLatestStates()
        {
            std::lock_guard<std::mutex> lock(mPropagate_);
            latest_time_ = Headers_[frame_count_];
            latest_P_ = Ps_[frame_count_];
            latest_Q_ = Eigen::Quaterniond(Rs_[frame_count_]);
            latest_V_ = Vs_[frame_count_];
            latest_Ba_ = Bas_[frame_count_];
            latest_Bg_ = Bgs_[frame_count_];
            latest_acc_0_ = acc_0_;
            latest_gyr_0_ = gyr_0_;

            backend_update_seq_.fetch_add(1, std::memory_order_release);
            if (marginalization_flag_ == MARGIN_OLD)
            {
                keyframe_seq_.fetch_add(1, std::memory_order_release);
            }
        }

        void Estimator::clearState()
        {
            {
                std::lock_guard lock(mBuf_);
                while (!acc_buf_.empty())
                    acc_buf_.pop();
                while (!gyr_buf_.empty())
                    gyr_buf_.pop();
                while (!feature_buf_.empty())
                    feature_buf_.pop();
            }
            buf_cv_.notify_all();

            prev_time_ = -1.0;
            cur_time_ = 0.0;
            init_first_pose_flag_ = false;
            first_imu_ = false;

            for (int i = 0; i < WINDOW_SIZE + 1; i++)
            {
                Rs_[i].setIdentity();
                Ps_[i].setZero();
                Vs_[i].setZero();
                Bas_[i].setZero();
                Bgs_[i].setZero();
                Headers_[i] = 0.0;

                dt_buf_[i].clear();
                linear_acceleration_buf_[i].clear();
                angular_velocity_buf_[i].clear();

                pre_integrations_[i].reset();
            }

            frame_count_ = 0;
            solver_flag_ = INITIAL;
            initialized_.store(false, std::memory_order_release);
            failure_last_P_valid_ = false;

            all_image_frame_.clear();
            tmp_pre_integration_.reset();

            feature_manager_->clearState();
            backend_->reset();

            {
                std::lock_guard<std::mutex> lock(mPropagate_);
                latest_time_ = 0.0;
                latest_P_.setZero();
                latest_V_.setZero();
                latest_Ba_.setZero();
                latest_Bg_.setZero();
                latest_acc_0_.setZero();
                latest_gyr_0_.setZero();
                latest_Q_.setIdentity();
            }
        }

        std::string Estimator::failureReasonName(uint8_t mask)
        {
            if (mask == 0)
                return "none";
            std::string s;
            auto add = [&](FailureReason bit, const char *name)
            {
                if (mask & static_cast<uint8_t>(bit))
                {
                    if (!s.empty())
                        s += " + ";
                    s += name;
                }
            };
            add(FailureReason::NaN, "NaN/Inf state");
            add(FailureReason::Velocity, "velocity blow-up");
            add(FailureReason::PositionJump, "position jump");
            add(FailureReason::AccelBias, "accel-bias blow-up");
            add(FailureReason::GyroBias, "gyro-bias blow-up");
            return s;
        }

        Estimator::FailureReport Estimator::failureDetection()
        {
            constexpr double kMaxVelocityNorm = 50.0; // m/s
            constexpr double kMaxPositionJump = 20.0; // m, per processed frame
            constexpr double kMaxAccelBias = 2.5;     // m/s^2
            constexpr double kMaxGyroBias = 1.0;      // rad/s
            const int wt = WINDOW_SIZE;
            const Eigen::Vector3d &P = Ps_[wt];
            const Eigen::Vector3d &V = Vs_[wt];
            const Eigen::Vector3d &Ba = Bas_[wt];
            const Eigen::Vector3d &Bg = Bgs_[wt];

            FailureReport r;

            // NaN/Inf -> unrecoverable.
            if (!P.allFinite() || !V.allFinite() || !Ba.allFinite() || !Bg.allFinite())
            {
                r.mask = static_cast<uint8_t>(FailureReason::NaN);
                r.detail = "non-finite pose/velocity/bias";
                return r;
            }

            const double jump = failure_last_P_valid_ ? (P - failure_last_P_).norm() : 0.0;
            if (V.norm() > kMaxVelocityNorm)
                r.mask |= static_cast<uint8_t>(FailureReason::Velocity);
            if (failure_last_P_valid_ && jump > kMaxPositionJump)
                r.mask |= static_cast<uint8_t>(FailureReason::PositionJump);
            if (Ba.norm() > kMaxAccelBias)
                r.mask |= static_cast<uint8_t>(FailureReason::AccelBias);
            if (Bg.norm() > kMaxGyroBias)
                r.mask |= static_cast<uint8_t>(FailureReason::GyroBias);

            if (r.mask != 0)
            {
                char buf[176];
                std::snprintf(buf, sizeof(buf),
                              "|V|=%.2f/%.0f |Ba|=%.3f/%.1f |Bg|=%.3f/%.1f jump=%.2f/%.0f",
                              V.norm(), kMaxVelocityNorm, Ba.norm(), kMaxAccelBias,
                              Bg.norm(), kMaxGyroBias, jump, kMaxPositionJump);
                r.detail = buf;
            }

            failure_last_P_ = P;
            failure_last_P_valid_ = true;
            return r;
        }

        void Estimator::slideWindow()
        {
            if (marginalization_flag_ == MARGIN_OLD)
            {
                slideWindowOld();
            }
            else
            {
                slideWindowNew();
            }
        }

        void Estimator::slideWindowOld()
        {
            // Remove oldest frame
            double t_0 = Headers_[0];

            // Transform IMU poses to camera poses for correct depth transfer
            Eigen::Matrix3d back_R0 = Rs_[0] * ric_[0];
            Eigen::Vector3d back_P0 = Ps_[0] + Rs_[0] * tic_[0];
            Eigen::Matrix3d new_R = Rs_[1] * ric_[0];
            Eigen::Vector3d new_P = Ps_[1] + Rs_[1] * tic_[0];

            if (all_image_frame_.count(t_0))
            {
                all_image_frame_.erase(t_0);
            }

            // Shift all states
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers_[i] = Headers_[i + 1];
                Rs_[i] = Rs_[i + 1];
                Ps_[i] = Ps_[i + 1];
                Vs_[i] = Vs_[i + 1];
                Bas_[i] = Bas_[i + 1];
                Bgs_[i] = Bgs_[i + 1];

                std::swap(pre_integrations_[i], pre_integrations_[i + 1]);

                dt_buf_[i].swap(dt_buf_[i + 1]);
                linear_acceleration_buf_[i].swap(linear_acceleration_buf_[i + 1]);
                angular_velocity_buf_[i].swap(angular_velocity_buf_[i + 1]);
            }

            // Reset newest slot
            Headers_[WINDOW_SIZE] = Headers_[WINDOW_SIZE - 1];
            Rs_[WINDOW_SIZE] = Rs_[WINDOW_SIZE - 1];
            Ps_[WINDOW_SIZE] = Ps_[WINDOW_SIZE - 1];
            Vs_[WINDOW_SIZE] = Vs_[WINDOW_SIZE - 1];
            Bas_[WINDOW_SIZE] = Bas_[WINDOW_SIZE - 1];
            Bgs_[WINDOW_SIZE] = Bgs_[WINDOW_SIZE - 1];

            pre_integrations_[WINDOW_SIZE].reset();
            dt_buf_[WINDOW_SIZE].clear();
            linear_acceleration_buf_[WINDOW_SIZE].clear();
            angular_velocity_buf_[WINDOW_SIZE].clear();

            if (solver_flag_ == NON_LINEAR)
                feature_manager_->removeBackShiftDepth(back_R0, back_P0, new_R, new_P);
            else
                feature_manager_->removeBack();
        }

        void Estimator::slideWindowNew()
        {
            // Remove second newest frame (keep newest)
            double t_margin = Headers_[WINDOW_SIZE - 1];

            if (all_image_frame_.count(t_margin))
            {
                all_image_frame_.erase(t_margin);
            }

            // Shift second newest to oldest
            Headers_[WINDOW_SIZE - 1] = Headers_[WINDOW_SIZE];
            Rs_[WINDOW_SIZE - 1] = Rs_[WINDOW_SIZE];
            Ps_[WINDOW_SIZE - 1] = Ps_[WINDOW_SIZE];
            Vs_[WINDOW_SIZE - 1] = Vs_[WINDOW_SIZE];
            Bas_[WINDOW_SIZE - 1] = Bas_[WINDOW_SIZE];
            Bgs_[WINDOW_SIZE - 1] = Bgs_[WINDOW_SIZE];

            // Merge pre-integrations
            for (size_t i = 0; i < dt_buf_[WINDOW_SIZE].size(); i++)
            {
                double tmp_dt = dt_buf_[WINDOW_SIZE][i];
                Eigen::Vector3d tmp_acc = linear_acceleration_buf_[WINDOW_SIZE][i];
                Eigen::Vector3d tmp_gyr = angular_velocity_buf_[WINDOW_SIZE][i];
                pre_integrations_[WINDOW_SIZE - 1]->push_back(tmp_dt, tmp_acc, tmp_gyr);

                dt_buf_[WINDOW_SIZE - 1].push_back(tmp_dt);
                linear_acceleration_buf_[WINDOW_SIZE - 1].push_back(tmp_acc);
                angular_velocity_buf_[WINDOW_SIZE - 1].push_back(tmp_gyr);
            }

            // Clear newest slot
            pre_integrations_[WINDOW_SIZE].reset();
            dt_buf_[WINDOW_SIZE].clear();
            linear_acceleration_buf_[WINDOW_SIZE].clear();
            angular_velocity_buf_[WINDOW_SIZE].clear();

            // Update feature manager
            feature_manager_->removeFront(frame_count_);
        }

        void Estimator::outliersRejection(std::set<int> &removeIndex)
        {
            const auto &features = feature_manager_->getFeatures();

            for (const auto &kv : features)
            {
                int feature_id = kv.first;
                const FeatureTrack &track = kv.second;

                if (track.estimated_depth <= 0)
                    continue;

                if (static_cast<int>(track.observations.size()) < 4)
                    continue;

                double depth = track.estimated_depth;
                int imu_i = track.start_frame;

                if (imu_i >= frame_count_)
                    continue;

                const Eigen::Vector3d &pts_i = track.observations[0].left().point_c;

                double err = 0;
                int errCnt = 0;

                for (size_t j = 1; j < track.observations.size(); j++)
                {
                    int imu_j = imu_i + static_cast<int>(j);
                    if (imu_j > frame_count_)
                        break;

                    const Eigen::Vector3d &pts_j = track.observations[j].left().point_c;

                    double tmp_error = reprojectionError(
                        Rs_[imu_i], Ps_[imu_i], ric_[0], tic_[0],
                        Rs_[imu_j], Ps_[imu_j], ric_[0], tic_[0],
                        depth, pts_i, pts_j);

                    err += tmp_error;
                    errCnt++;

                    // Also check right camera if stereo
                    if (track.observations[j].is_stereo)
                    {
                        const Eigen::Vector3d &pts_j_right = track.observations[j].right().point_c;
                        double tmp_error_r = reprojectionError(
                            Rs_[imu_i], Ps_[imu_i], ric_[0], tic_[0],
                            Rs_[imu_j], Ps_[imu_j], ric_[1], tic_[1],
                            depth, pts_i, pts_j_right);
                        err += tmp_error_r;
                        errCnt++;
                    }
                }

                if (errCnt > 0)
                {
                    double ave_err = err / errCnt;
                    if (ave_err * config_.focal_length > config_.max_reprojection_error)
                    {
                        removeIndex.insert(feature_id);
                    }
                }
            }
        }

        double Estimator::reprojectionError(const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Pi,
                                            const Eigen::Matrix3d &rici, const Eigen::Vector3d &tici,
                                            const Eigen::Matrix3d &Rj, const Eigen::Vector3d &Pj,
                                            const Eigen::Matrix3d &ricj, const Eigen::Vector3d &ticj,
                                            double depth, const Eigen::Vector3d &uvi, const Eigen::Vector3d &uvj)
        {
            // Transform point from camera i to world
            Eigen::Vector3d pts_camera_i = uvi * depth;
            Eigen::Vector3d pts_imu_i = rici * pts_camera_i + tici;
            Eigen::Vector3d pts_world = Ri * pts_imu_i + Pi;

            // Transform point from world to camera j
            Eigen::Vector3d pts_imu_j = Rj.transpose() * (pts_world - Pj);
            Eigen::Vector3d pts_camera_j = ricj.transpose() * (pts_imu_j - ticj);

            double dep_j = pts_camera_j.z();
            double rx = (pts_camera_j.x() / dep_j) - uvj.x();
            double ry = (pts_camera_j.y() / dep_j) - uvj.y();

            return std::sqrt(rx * rx + ry * ry);
        }

        bool Estimator::relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                auto corres = feature_manager_->getCorresponding(i, WINDOW_SIZE);
                if (static_cast<int>(corres.size()) > 20)
                {
                    double sum_parallax = 0;
                    for (const auto &c : corres)
                    {
                        Eigen::Vector2d pts_0(c.first(0), c.first(1));
                        Eigen::Vector2d pts_1(c.second(0), c.second(1));
                        sum_parallax += (pts_0 - pts_1).norm();
                    }
                    double average_parallax = sum_parallax / static_cast<int>(corres.size());
                    if (average_parallax > config_.init_min_parallax_ratio &&
                        motion_estimator_.solveRelativeRT(corres, relative_R, relative_T))
                    {
                        l = i;
                        RCLCPP_DEBUG(rclcpp::get_logger("estimator"),
                                     "relativePose: parallax=%.1f px@ref (gate=%.1f), frame l=%d",
                                     average_parallax * REF_FOCAL_PX,
                                     config_.init_min_parallax_ratio * REF_FOCAL_PX, l);
                        return true;
                    }
                }
            }
            return false;
        }

        bool Estimator::initialStructure()
        {
            // Apply static-period Bgs prior before alignment when the IMU window flagged a
            // low-variance prefix. Empirical: load-bearing for V1_03 LK methods.
            if (bootstrap_static_seen_)
            {
                for (int i = 0; i <= WINDOW_SIZE; i++)
                    Bgs_[i] = bootstrap_gyro_mean_;
            }

            // Build SFMFeature list from feature manager
            const auto &features = feature_manager_->getFeatures();
            std::vector<SFMFeature> sfm_f;
            sfm_f.reserve(features.size());
            for (const auto &[feature_id, track] : features)
            {
                SFMFeature tmp_feature;
                tmp_feature.state = false;
                tmp_feature.id = feature_id;
                int imu_j = track.start_frame - 1;
                for (const auto &obs : track.observations)
                {
                    imu_j++;
                    tmp_feature.observation.emplace_back(
                        imu_j, Eigen::Vector2d{obs.left().point_c.x(), obs.left().point_c.y()});
                }
                sfm_f.push_back(std::move(tmp_feature));
            }

            // Find relative pose
            Eigen::Matrix3d relative_R;
            Eigen::Vector3d relative_T;
            int l;
            if (!relativePose(relative_R, relative_T, l))
            {
                RCLCPP_INFO(rclcpp::get_logger("estimator"),
                            "Not enough features or parallax; Move device around");
                return false;
            }

            // Global SFM
            std::vector<Eigen::Quaterniond> Q(frame_count_ + 1);
            std::vector<Eigen::Vector3d> T(frame_count_ + 1);
            std::map<int, Eigen::Vector3d> sfm_tracked_points;
            GlobalSFM sfm;
            if (!sfm.construct(frame_count_ + 1, Q.data(), T.data(), l,
                               relative_R, relative_T, sfm_f, sfm_tracked_points))
            {
                RCLCPP_DEBUG(rclcpp::get_logger("estimator"), "Global SFM failed!");
                marginalization_flag_ = MARGIN_OLD;
                return false;
            }

            // Solve PnP for all frames in all_image_frame_ (including non-keyframes)
            auto frame_it = all_image_frame_.begin();
            for (int i = 0; frame_it != all_image_frame_.end(); ++frame_it)
            {
                if (frame_it->first == Headers_[i])
                {
                    frame_it->second.is_key_frame = true;
                    frame_it->second.R = Q[i].toRotationMatrix() * ric_[0].transpose();
                    frame_it->second.T = T[i];
                    i++;
                    continue;
                }
                if (frame_it->first > Headers_[i])
                    i++;

                Eigen::Matrix3d R_initial = (Q[i].inverse()).toRotationMatrix();
                Eigen::Vector3d P_initial = -R_initial * T[i];
                cv::Mat tmp_r, rvec, t, D;
                cv::eigen2cv(R_initial, tmp_r);
                cv::Rodrigues(tmp_r, rvec);
                cv::eigen2cv(P_initial, t);

                frame_it->second.is_key_frame = false;
                std::vector<cv::Point3f> pts_3_vector;
                std::vector<cv::Point2f> pts_2_vector;

                // Build 3D-2D correspondences from feature observations at this frame
                for (const auto &[feature_id, track] : features)
                {
                    auto sfm_it = sfm_tracked_points.find(feature_id);
                    if (sfm_it == sfm_tracked_points.end())
                        continue;
                    // Check if this feature is observed at the current non-keyframe timestamp
                    int obs_frame = track.start_frame;
                    for (const auto &obs : track.observations)
                    {
                        // Match by checking if obs_frame corresponds to current iterator frame
                        // For non-keyframes we check timestamp proximity
                        if (obs_frame >= 0 && obs_frame <= frame_count_)
                        {
                            // We need observation at this specific frame - use header matching
                            if (std::abs(Headers_[obs_frame] - frame_it->first) < 1e-6)
                            {
                                Eigen::Vector3d world_pts = sfm_it->second;
                                pts_3_vector.emplace_back(
                                    static_cast<float>(world_pts(0)),
                                    static_cast<float>(world_pts(1)),
                                    static_cast<float>(world_pts(2)));
                                pts_2_vector.emplace_back(
                                    static_cast<float>(obs.left().point_c.x()),
                                    static_cast<float>(obs.left().point_c.y()));
                                break;
                            }
                        }
                        obs_frame++;
                    }
                }

                cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
                if (static_cast<int>(pts_3_vector.size()) < 6)
                {
                    RCLCPP_DEBUG(rclcpp::get_logger("estimator"),
                                 "Not enough points for PnP (%zu)", pts_3_vector.size());
                    return false;
                }
                if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true))
                {
                    RCLCPP_DEBUG(rclcpp::get_logger("estimator"), "solvePnP failed");
                    return false;
                }
                cv::Rodrigues(rvec, tmp_r);
                Eigen::MatrixXd R_pnp, T_pnp;
                cv::cv2eigen(tmp_r, R_pnp);
                cv::cv2eigen(t, T_pnp);
                Eigen::Matrix3d R_pnp_t = R_pnp.transpose();
                Eigen::Vector3d T_pnp_w = R_pnp_t * (-T_pnp);
                frame_it->second.R = R_pnp_t * ric_[0].transpose();
                frame_it->second.T = T_pnp_w;
            }

            return visualInitialAlign();
        }

        bool Estimator::visualInitialAlign()
        {
            Eigen::VectorXd x;
            Eigen::Vector3d g;

            // Full visual-IMU alignment: gyro bias + velocity + gravity + scale
            bool result = VisualIMUAlignment::solveVelocityGravityScale(
                all_image_frame_, Bgs_.data(), g, x,
                config_.gravity_magnitude, tic_[0], WINDOW_SIZE,
                config_.init_max_g_error);

            if (!result)
            {
                RCLCPP_DEBUG(rclcpp::get_logger("estimator"), "Visual-IMU alignment failed!");
                return false;
            }

            // Copy aligned poses to state arrays
            for (int i = 0; i <= frame_count_; i++)
            {
                Eigen::Matrix3d Ri = all_image_frame_[Headers_[i]].R;
                Eigen::Vector3d Pi = all_image_frame_[Headers_[i]].T;
                Ps_[i] = Pi;
                Rs_[i] = Ri;
                all_image_frame_[Headers_[i]].is_key_frame = true;
            }

            // Extract scale
            double s = (x.tail<1>())(0);
            init_scale_ = s;
            init_g_norm_ = g.norm();

            // Repropagate pre-integrations with corrected gyro bias
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (pre_integrations_[i])
                    pre_integrations_[i]->repropagate(Eigen::Vector3d::Zero(), Bgs_[i]);
            }

            // Apply scale and remove camera extrinsic offset
            for (int i = frame_count_; i >= 0; i--)
                Ps_[i] = s * Ps_[i] - Rs_[i] * tic_[0] - (s * Ps_[0] - Rs_[0] * tic_[0]);

            // Initialize velocities from alignment solution
            int kv = -1;
            for (auto &[timestamp, frame] : all_image_frame_)
            {
                if (frame.is_key_frame)
                {
                    kv++;
                    Vs_[kv] = frame.R * x.segment<3>(kv * 3);
                }
            }

            // Rotate everything to gravity-aligned world frame
            Eigen::Matrix3d R0 = Utility::g2R(g);
            double yaw = Utility::R2ypr(R0 * Rs_[0]).x();
            R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
            g = R0 * g;

            Eigen::Matrix3d rot_diff = R0;
            for (int i = 0; i <= frame_count_; i++)
            {
                Ps_[i] = rot_diff * Ps_[i];
                Rs_[i] = rot_diff * Rs_[i];
                Vs_[i] = rot_diff * Vs_[i];
            }

            RCLCPP_DEBUG(rclcpp::get_logger("estimator"),
                         "Aligned gravity: [%.4f, %.4f, %.4f] |g|=%.4f",
                         g.x(), g.y(), g.z(), g.norm());

            // Update gravity_ to the refined direction
            gravity_ = g;

            // Re-triangulate features at correct scale
            feature_manager_->clearDepth();
            feature_manager_->triangulate(frame_count_, Ps_.data(), Rs_.data(),
                                          tic_.data(), ric_.data());

            return true;
        }

    } // namespace perception
} // namespace uosm
