#include "../../include/backend/ceres_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <rclcpp/logging.hpp>
#include <thread>

namespace
{
    // Observability prior on a camera extrinsic block [tx,ty,tz, qx,qy,qz,qw].
    struct ExtrinsicPriorFunctor
    {
        ExtrinsicPriorFunctor(const Eigen::Vector3d &t_ref, const Eigen::Quaterniond &q_ref,
                              double w_t, double w_r)
            : t_ref_(t_ref), q_ref_(q_ref), w_t_(w_t), w_r_(w_r) {}

        template <typename T>
        bool operator()(const T *const ex, T *res) const
        {
            res[0] = w_t_ * (ex[0] - T(t_ref_.x()));
            res[1] = w_t_ * (ex[1] - T(t_ref_.y()));
            res[2] = w_t_ * (ex[2] - T(t_ref_.z()));
            // storage is [x,y,z,w]; Eigen::Quaternion ctor is (w,x,y,z)
            Eigen::Quaternion<T> q(ex[6], ex[3], ex[4], ex[5]);
            Eigen::Quaternion<T> q_ref_inv(T(q_ref_.w()), T(-q_ref_.x()), T(-q_ref_.y()), T(-q_ref_.z()));
            Eigen::Quaternion<T> dq = q_ref_inv * q;
            const T sign = dq.w() < T(0) ? T(-1) : T(1); // resolve double-cover
            res[3] = w_r_ * T(2) * sign * dq.x();
            res[4] = w_r_ * T(2) * sign * dq.y();
            res[5] = w_r_ * T(2) * sign * dq.z();
            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d &t_ref,
                                           const Eigen::Quaterniond &q_ref,
                                           double w_t, double w_r)
        {
            return new ceres::AutoDiffCostFunction<ExtrinsicPriorFunctor, 6, 7>(
                new ExtrinsicPriorFunctor(t_ref, q_ref, w_t, w_r));
        }

        Eigen::Vector3d t_ref_;
        Eigen::Quaterniond q_ref_;
        double w_t_, w_r_;
    };
} // namespace

namespace uosm
{
    namespace perception
    {

        CeresBackend::CeresBackend(const BackendSolverConfig &config)
            : config_(config)
        {
            initParameters();
        }

        void CeresBackend::initParameters()
        {
            const int n = WINDOW_SIZE + 1;
            para_Pose_.resize(n);
            para_SpeedBias_.resize(n);
            for (int i = 0; i < n; i++)
            {
                para_Pose_[i].fill(0.0);
                para_SpeedBias_[i].fill(0.0);
            }

            for (auto &ex : para_Ex_Pose_)
            {
                ex.fill(0.0);
                ex[6] = 1.0; // identity quaternion w
            }

            para_Td_[0] = config_.initial_td;
            para_Feature_.clear();
        }

        void CeresBackend::reset()
        {
            initParameters();
            last_marginalization_info_.reset();
            last_marginalization_parameter_blocks_.clear();
            stats_ = SolverStatistics();
            open_ex_estimation_ = false;
        }

        void CeresBackend::vector2double(int frame_count,
                                         Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs,
                                         Eigen::Vector3d *Vs, Eigen::Vector3d *Bas, Eigen::Vector3d *Bgs,
                                         Eigen::Vector3d *tic, Eigen::Matrix3d *ric)
        {
            for (int i = 0; i <= frame_count; i++)
            {
                para_Pose_[i][0] = Ps[i].x();
                para_Pose_[i][1] = Ps[i].y();
                para_Pose_[i][2] = Ps[i].z();

                Eigen::Quaterniond q(Rs[i]);
                para_Pose_[i][3] = q.x();
                para_Pose_[i][4] = q.y();
                para_Pose_[i][5] = q.z();
                para_Pose_[i][6] = q.w();

                para_SpeedBias_[i][0] = Vs[i].x();
                para_SpeedBias_[i][1] = Vs[i].y();
                para_SpeedBias_[i][2] = Vs[i].z();

                para_SpeedBias_[i][3] = Bas[i].x();
                para_SpeedBias_[i][4] = Bas[i].y();
                para_SpeedBias_[i][5] = Bas[i].z();

                para_SpeedBias_[i][6] = Bgs[i].x();
                para_SpeedBias_[i][7] = Bgs[i].y();
                para_SpeedBias_[i][8] = Bgs[i].z();
            }

            for (int c = 0; c < 2; c++)
            {
                para_Ex_Pose_[c][0] = tic[c].x();
                para_Ex_Pose_[c][1] = tic[c].y();
                para_Ex_Pose_[c][2] = tic[c].z();
                Eigen::Quaterniond q_ic(ric[c]);
                para_Ex_Pose_[c][3] = q_ic.x();
                para_Ex_Pose_[c][4] = q_ic.y();
                para_Ex_Pose_[c][5] = q_ic.z();
                para_Ex_Pose_[c][6] = q_ic.w();
            }
        }

        void CeresBackend::double2vector(int frame_count,
                                         Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs,
                                         Eigen::Vector3d *Vs, Eigen::Vector3d *Bas, Eigen::Vector3d *Bgs,
                                         Eigen::Vector3d *tic, Eigen::Matrix3d *ric)
        {
            // Yaw alignment: compensate the yaw gauge freedom so the
            // marginalization prior's linearization point stays consistent.
            Eigen::Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
            Eigen::Vector3d origin_P0 = Ps[0];

            Eigen::Vector3d origin_R00 = Utility::R2ypr(
                Eigen::Quaterniond(para_Pose_[0][6], para_Pose_[0][3],
                                   para_Pose_[0][4], para_Pose_[0][5])
                    .toRotationMatrix());
            double y_diff = origin_R0.x() - origin_R00.x();

            Eigen::Matrix3d rot_diff = Utility::ypr2R(Eigen::Vector3d(y_diff, 0, 0));
            if (std::abs(std::abs(origin_R0.y()) - 90) < 1.0 ||
                std::abs(std::abs(origin_R00.y()) - 90) < 1.0)
            {
                rot_diff = Rs[0] * Eigen::Quaterniond(para_Pose_[0][6], para_Pose_[0][3],
                                                      para_Pose_[0][4], para_Pose_[0][5])
                                       .toRotationMatrix()
                                       .transpose();
            }

            for (int i = 0; i <= frame_count; i++)
            {
                Rs[i] = rot_diff * Eigen::Quaterniond(para_Pose_[i][6], para_Pose_[i][3],
                                                      para_Pose_[i][4], para_Pose_[i][5])
                                       .normalized()
                                       .toRotationMatrix();

                Ps[i] = rot_diff * Eigen::Vector3d(para_Pose_[i][0] - para_Pose_[0][0],
                                                   para_Pose_[i][1] - para_Pose_[0][1],
                                                   para_Pose_[i][2] - para_Pose_[0][2]) +
                        origin_P0;

                Vs[i] = rot_diff * Eigen::Vector3d(para_SpeedBias_[i][0],
                                                   para_SpeedBias_[i][1],
                                                   para_SpeedBias_[i][2]);

                Bas[i] = Eigen::Vector3d(para_SpeedBias_[i][3],
                                         para_SpeedBias_[i][4],
                                         para_SpeedBias_[i][5]);

                Bgs[i] = Eigen::Vector3d(para_SpeedBias_[i][6],
                                         para_SpeedBias_[i][7],
                                         para_SpeedBias_[i][8]);
            }

            if (config_.estimate_extrinsic)
            {
                for (int c = 0; c < 2; c++)
                {
                    tic[c] = Eigen::Vector3d(para_Ex_Pose_[c][0],
                                             para_Ex_Pose_[c][1],
                                             para_Ex_Pose_[c][2]);
                    ric[c] = Eigen::Quaterniond(para_Ex_Pose_[c][6], para_Ex_Pose_[c][3],
                                                para_Ex_Pose_[c][4], para_Ex_Pose_[c][5])
                                 .normalized()
                                 .toRotationMatrix();
                }
            }
        }

        void CeresBackend::addIMUResidualBlocks(ceres::Problem &problem,
                                                int frame_count,
                                                std::vector<std::unique_ptr<IntegrationBase>> &pre_integrations,
                                                std::vector<ceres::ResidualBlockId> *imu_ids)
        {
            int imu_factors = 0;
            for (int i = 0; i < frame_count; i++)
            {
                if (pre_integrations[i + 1] && pre_integrations[i + 1]->sum_dt > 0)
                {
                    if (pre_integrations[i + 1]->sum_dt > config_.max_imu_sum_dt)
                    {
                        RCLCPP_DEBUG(rclcpp::get_logger("ceres_backend"),
                                     "Skipping IMU factor %d due to long sum_dt: %fs",
                                     i, pre_integrations[i + 1]->sum_dt);
                        continue;
                    }

                    auto *imu_factor = new IMUFactor(pre_integrations[i + 1].get());
                    ceres::ResidualBlockId id = problem.AddResidualBlock(
                        imu_factor, nullptr,
                        para_Pose_[i].data(), para_SpeedBias_[i].data(),
                        para_Pose_[i + 1].data(), para_SpeedBias_[i + 1].data());
                    if (imu_ids)
                        imu_ids->push_back(id);
                    imu_factors++;
                }
            }
            RCLCPP_DEBUG(rclcpp::get_logger("ceres_backend"), "Added %d IMU factors", imu_factors);
        }

        void CeresBackend::addVisualResidualBlocks(ceres::Problem &problem,
                                                   int frame_count,
                                                   std::shared_ptr<FeatureManager> feature_manager,
                                                   ReprojBlockIds *reproj_ids)
        {
            if (!feature_manager)
                return;

            const auto &features = feature_manager->getFeatures();

            for (const auto &[feature_id, track] : features)
            {
                if (track.estimated_depth <= 0)
                {
                    continue;
                }

                if (static_cast<int>(track.observations.size()) < config_.min_observations)
                {
                    continue;
                }

                int imu_i = track.start_frame;
                if (imu_i > frame_count)
                    continue;

                auto [it, inserted] = para_Feature_.try_emplace(feature_id);
                if (inserted)
                    it->second[0] = 1.0 / track.estimated_depth;
                double *para_inv_depth = it->second.data();

                ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

                const auto &obs_i = track.observations[0].left();
                Eigen::Vector3d pts_i(obs_i.point_c.x(), obs_i.point_c.y(), 1.0);
                Eigen::Vector2d velocity_i(obs_i.velocity.x(), obs_i.velocity.y());
                double td_i = obs_i.cur_td;

                if (track.observations[0].is_stereo)
                {
                    const auto &obs_i_right = track.observations[0].right();
                    Eigen::Vector3d pts_i_right(obs_i_right.point_c.x(), obs_i_right.point_c.y(), 1.0);
                    Eigen::Vector2d velocity_i_right(obs_i_right.velocity.x(), obs_i_right.velocity.y());

                    auto *f = new ProjectionOneFrameTwoCamFactor(
                        pts_i, pts_i_right, velocity_i, velocity_i_right, td_i, td_i, config_.focal_length);

                    ceres::ResidualBlockId id = problem.AddResidualBlock(
                        f, loss_function,
                        para_Ex_Pose_[0].data(), para_Ex_Pose_[1].data(),
                        para_inv_depth, para_Td_.data());
                    if (reproj_ids)
                        reproj_ids->stereo.push_back(id);
                }

                for (size_t k = 1; k < track.observations.size(); k++)
                {
                    int imu_j = imu_i + static_cast<int>(k);
                    if (imu_j > frame_count)
                        break;

                    const auto &obs_j_left = track.observations[k].left();
                    Eigen::Vector3d pts_j(obs_j_left.point_c.x(), obs_j_left.point_c.y(), 1.0);
                    Eigen::Vector2d velocity_j(obs_j_left.velocity.x(), obs_j_left.velocity.y());
                    double td_j = obs_j_left.cur_td;

                    auto *f_mono = new ProjectionTwoFrameOneCamFactor(
                        pts_i, pts_j, velocity_i, velocity_j, td_i, td_j,
                        config_.focal_length, obs_j_left.has_cov, obs_j_left.sqrt_info_px);

                    ceres::ResidualBlockId id_mono = problem.AddResidualBlock(
                        f_mono, loss_function,
                        para_Pose_[imu_i].data(), para_Pose_[imu_j].data(),
                        para_Ex_Pose_[0].data(), para_inv_depth, para_Td_.data());
                    if (reproj_ids)
                        reproj_ids->temporal.push_back(id_mono);

                    if (track.observations[k].is_stereo)
                    {
                        const auto &obs_j_right = track.observations[k].right();
                        Eigen::Vector3d pts_j_right(obs_j_right.point_c.x(), obs_j_right.point_c.y(), 1.0);
                        Eigen::Vector2d velocity_j_right(obs_j_right.velocity.x(), obs_j_right.velocity.y());

                        auto *f_stereo = new ProjectionTwoFrameTwoCamFactor(
                            pts_i, pts_j_right, velocity_i, velocity_j_right, td_i, td_j,
                            config_.focal_length, obs_j_left.has_cov, obs_j_left.sqrt_info_px);

                        ceres::ResidualBlockId id_stereo = problem.AddResidualBlock(
                            f_stereo, loss_function,
                            para_Pose_[imu_i].data(), para_Pose_[imu_j].data(),
                            para_Ex_Pose_[0].data(), para_Ex_Pose_[1].data(),
                            para_inv_depth, para_Td_.data());
                        if (reproj_ids)
                            reproj_ids->cross_cam.push_back(id_stereo);
                    }
                }
            }
        }

        void CeresBackend::addMarginalizationResidualBlock(ceres::Problem &problem,
                                                           std::vector<ceres::ResidualBlockId> *marg_ids)
        {
            if (last_marginalization_info_ && last_marginalization_info_->valid)
            {
                auto *marginalization_factor =
                    new MarginalizationFactor(last_marginalization_info_.get());

                ceres::ResidualBlockId id = problem.AddResidualBlock(
                    marginalization_factor, nullptr,
                    last_marginalization_parameter_blocks_);
                if (marg_ids)
                    marg_ids->push_back(id);
            }
        }

        bool CeresBackend::solve(
            int frame_count,
            Eigen::Vector3d *Ps,
            Eigen::Matrix3d *Rs,
            Eigen::Vector3d *Vs,
            Eigen::Vector3d *Bas,
            Eigen::Vector3d *Bgs,
            Eigen::Vector3d *tic,
            Eigen::Matrix3d *ric,
            std::vector<std::unique_ptr<IntegrationBase>> &pre_integrations,
            std::shared_ptr<FeatureManager> feature_manager,
            int marginalization_flag)
        {
            auto solve_start = std::chrono::high_resolution_clock::now();

            vector2double(frame_count, Ps, Rs, Vs, Bas, Bgs, tic, ric);

            if (!open_ex_estimation_)
            {
                for (int c = 0; c < 2; ++c)
                {
                    ex_ref_t_[c] = tic[c];
                    ex_ref_q_[c] = Eigen::Quaterniond(ric[c]);
                }
            }

            ceres::Problem problem;

            for (int i = 0; i <= frame_count; i++)
            {
                problem.AddParameterBlock(para_Pose_[i].data(), 7, new PoseManifold());
                problem.AddParameterBlock(para_SpeedBias_[i].data(), 9);
            }

            problem.AddParameterBlock(para_Ex_Pose_[0].data(), 7, new PoseManifold());
            problem.AddParameterBlock(para_Ex_Pose_[1].data(), 7, new PoseManifold());
            if (config_.estimate_extrinsic &&
                ((frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || open_ex_estimation_))
            {
                open_ex_estimation_ = true;
                constexpr double kSigmaT = 0.1;                // m
                constexpr double kSigmaR = 2.0 * M_PI / 180.0; // rad
                for (int c = 0; c < 2; ++c)
                    problem.AddResidualBlock(
                        ExtrinsicPriorFunctor::Create(ex_ref_t_[c], ex_ref_q_[c],
                                                      1.0 / kSigmaT, 1.0 / kSigmaR),
                        nullptr, para_Ex_Pose_[c].data());
            }
            else
            {
                problem.SetParameterBlockConstant(para_Ex_Pose_[0].data());
                problem.SetParameterBlockConstant(para_Ex_Pose_[1].data());
            }

            problem.AddParameterBlock(para_Td_.data(), 1);
            // Holds td constant until there is motion excitation.
            if (!config_.estimate_td || Vs[0].norm() < 0.2)
                problem.SetParameterBlockConstant(para_Td_.data());

            std::vector<ceres::ResidualBlockId> marg_ids, imu_ids;
            ReprojBlockIds reproj_ids;
            addMarginalizationResidualBlock(problem, &marg_ids);
            addIMUResidualBlocks(problem, frame_count, pre_integrations, &imu_ids);
            addVisualResidualBlocks(problem, frame_count, feature_manager, &reproj_ids);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            if (config_.use_gpu)
                options.dense_linear_algebra_library_type = ceres::CUDA;
            options.trust_region_strategy_type = ceres::DOGLEG;
            options.max_num_iterations = config_.max_iterations;
            const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
            options.num_threads = (config_.num_threads > 0)
                                      ? config_.num_threads
                                      : std::max(1, hw_threads - 2);
            if (marginalization_flag == 0) // MARGIN_OLD
                options.max_solver_time_in_seconds = config_.max_solver_time * 0.8;
            else
                options.max_solver_time_in_seconds = config_.max_solver_time;
            options.minimizer_progress_to_stdout = false;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            auto solve_end = std::chrono::high_resolution_clock::now();

            stats_.final_cost = summary.final_cost;
            stats_.iterations = static_cast<int>(summary.iterations.size());
            stats_.converged = (summary.termination_type == ceres::CONVERGENCE);
            stats_.solve_time_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
            stats_.solver_info = summary.BriefReport();

            // Cost contribution per factor group at the converged point (robustified,
            // matching summary.final_cost = 0.5 * sum rho(r^2)). Splits reprojection into:
            //   temporal  - ProjectionTwoFrameOneCamFactor  (left cam across two frames)
            //   stereo    - ProjectionOneFrameTwoCamFactor  (left/right pair, same frame)
            //   cross_cam - ProjectionTwoFrameTwoCamFactor  (left@i vs right@j, across frames)
            auto group_cost = [&problem](const std::vector<ceres::ResidualBlockId> &ids) -> double
            {
                if (ids.empty())
                    return 0.0;
                ceres::Problem::EvaluateOptions opts;
                opts.residual_blocks = ids;
                opts.apply_loss_function = true;
                double cost = 0.0;
                problem.Evaluate(opts, &cost, nullptr, nullptr, nullptr);
                return cost;
            };
            stats_.cost_marg = group_cost(marg_ids);
            stats_.cost_imu = group_cost(imu_ids);
            stats_.cost_temporal = group_cost(reproj_ids.temporal);
            stats_.cost_stereo = group_cost(reproj_ids.stereo);
            stats_.cost_cross_cam = group_cost(reproj_ids.cross_cam);

            RCLCPP_DEBUG(rclcpp::get_logger("ceres_backend"),
                         "Solve: %.1fms iter=%d %s",
                         stats_.solve_time_ms, stats_.iterations,
                         stats_.converged ? "CONVERGED" : "NO_CONV");

            double2vector(frame_count, Ps, Rs, Vs, Bas, Bgs, tic, ric);

            if (feature_manager)
            {
                auto &features = const_cast<std::map<int, FeatureTrack> &>(feature_manager->getFeatures());
                for (auto it = features.begin(); it != features.end();)
                {
                    auto fit = para_Feature_.find(it->first);
                    if (fit != para_Feature_.end())
                    {
                        const double inv_depth = fit->second[0];
                        // Cull failures (VINS-Fusion setDepth(solve_flag=2) + removeFailures()),
                        const bool cull = (inv_depth <= 1.0 / MAX_DEPTH || inv_depth >= 1.0 / MIN_DEPTH);
                        if (cull)
                        {
                            para_Feature_.erase(fit);
                            it = features.erase(it);
                            continue;
                        }
                        it->second.estimated_depth = 1.0 / inv_depth;
                        it->second.has_valid_depth = true;
                    }
                    ++it;
                }
            }

            return stats_.converged;
        }

        void CeresBackend::marginalize(
            int frame_count,
            Eigen::Vector3d *Ps,
            Eigen::Matrix3d *Rs,
            Eigen::Vector3d *Vs,
            Eigen::Vector3d *Bas,
            Eigen::Vector3d *Bgs,
            Eigen::Vector3d *tic,
            Eigen::Matrix3d *ric,
            std::vector<std::unique_ptr<IntegrationBase>> &pre_integrations,
            std::shared_ptr<FeatureManager> feature_manager,
            bool marginalize_old)
        {
            auto marg_start = std::chrono::high_resolution_clock::now();

            if (marginalize_old)
            {
                marginalizeOldFrame(frame_count, Ps, Rs, Vs, Bas, Bgs, tic, ric,
                                    pre_integrations, feature_manager);
            }
            else
            {
                marginalizeNewFrame(frame_count, Ps, Rs, Vs, Bas, Bgs, tic, ric);
            }

            auto marg_end = std::chrono::high_resolution_clock::now();
            stats_.marginalization_time_ms = std::chrono::duration<double, std::milli>(marg_end - marg_start).count();
        }

        void CeresBackend::marginalizeOldFrame(int frame_count,
                                               Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs,
                                               Eigen::Vector3d *Vs, Eigen::Vector3d *Bas, Eigen::Vector3d *Bgs,
                                               Eigen::Vector3d *tic, Eigen::Matrix3d *ric,
                                               std::vector<std::unique_ptr<IntegrationBase>> &pre_integrations,
                                               std::shared_ptr<FeatureManager> feature_manager)
        {
            auto marginalization_info = std::make_unique<MarginalizationInfo>();

            vector2double(frame_count, Ps, Rs, Vs, Bas, Bgs, tic, ric);

            if (last_marginalization_info_ && last_marginalization_info_->valid)
            {
                std::vector<int> drop_set;
                for (size_t i = 0; i < last_marginalization_parameter_blocks_.size(); i++)
                {
                    if (last_marginalization_parameter_blocks_[i] == para_Pose_[0].data() ||
                        last_marginalization_parameter_blocks_[i] == para_SpeedBias_[0].data())
                    {
                        drop_set.push_back(static_cast<int>(i));
                    }
                }

                auto *marginalization_factor =
                    new MarginalizationFactor(last_marginalization_info_.get());

                auto *residual_block_info = new ResidualBlockInfo(
                    marginalization_factor, nullptr,
                    last_marginalization_parameter_blocks_, drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            if (pre_integrations[1] && pre_integrations[1]->sum_dt > 0 &&
                pre_integrations[1]->sum_dt < config_.max_imu_sum_dt)
            {
                auto *imu_factor = new IMUFactor(pre_integrations[1].get());

                auto *residual_block_info = new ResidualBlockInfo(
                    imu_factor, nullptr,
                    std::vector<double *>{para_Pose_[0].data(), para_SpeedBias_[0].data(),
                                          para_Pose_[1].data(), para_SpeedBias_[1].data()},
                    std::vector<int>{0, 1});

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            if (feature_manager)
            {
                const auto &features = feature_manager->getFeatures();
                const int min_obs_mature = config_.min_observations;

                for (const auto &[feature_id, track] : features)
                {
                    if (track.start_frame != 0 || track.estimated_depth <= 0)
                        continue;

                    if (static_cast<int>(track.observations.size()) < min_obs_mature)
                        continue;

                    auto fit = para_Feature_.find(feature_id);
                    if (fit == para_Feature_.end())
                        continue;
                    double *para_inv_depth = fit->second.data();

                    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

                    const auto &obs_i = track.observations[0].left();
                    Eigen::Vector3d pts_i(obs_i.point_c.x(), obs_i.point_c.y(), 1.0);
                    Eigen::Vector2d velocity_i(obs_i.velocity.x(), obs_i.velocity.y());
                    double td_i = obs_i.cur_td;

                    if (track.observations[0].is_stereo)
                    {
                        const auto &obs_i_right = track.observations[0].right();
                        Eigen::Vector3d pts_i_right(obs_i_right.point_c.x(), obs_i_right.point_c.y(), 1.0);
                        Eigen::Vector2d velocity_i_right(obs_i_right.velocity.x(), obs_i_right.velocity.y());

                        auto *f_stereo = new ProjectionOneFrameTwoCamFactor(
                            pts_i, pts_i_right, velocity_i, velocity_i_right, td_i, td_i, config_.focal_length);

                        auto *rbi = new ResidualBlockInfo(
                            f_stereo, loss_function,
                            std::vector<double *>{para_Ex_Pose_[0].data(), para_Ex_Pose_[1].data(),
                                                  para_inv_depth, para_Td_.data()},
                            std::vector<int>{2});

                        marginalization_info->addResidualBlockInfo(rbi);
                    }

                    for (size_t k = 1; k < track.observations.size(); k++)
                    {
                        int imu_j = track.start_frame + static_cast<int>(k);
                        if (imu_j > frame_count)
                            break;

                        const auto &obs_j_left = track.observations[k].left();
                        Eigen::Vector3d pts_j(obs_j_left.point_c.x(), obs_j_left.point_c.y(), 1.0);
                        Eigen::Vector2d velocity_j(obs_j_left.velocity.x(), obs_j_left.velocity.y());
                        double td_j = obs_j_left.cur_td;

                        auto *f_mono = new ProjectionTwoFrameOneCamFactor(
                            pts_i, pts_j, velocity_i, velocity_j, td_i, td_j,
                            config_.focal_length, obs_j_left.has_cov, obs_j_left.sqrt_info_px);

                        auto *rbi_mono = new ResidualBlockInfo(
                            f_mono, loss_function,
                            std::vector<double *>{para_Pose_[0].data(), para_Pose_[imu_j].data(),
                                                  para_Ex_Pose_[0].data(), para_inv_depth, para_Td_.data()},
                            std::vector<int>{0, 3});

                        marginalization_info->addResidualBlockInfo(rbi_mono);

                        if (track.observations[k].is_stereo)
                        {
                            const auto &obs_j_right = track.observations[k].right();
                            Eigen::Vector3d pts_j_right(obs_j_right.point_c.x(), obs_j_right.point_c.y(), 1.0);
                            Eigen::Vector2d velocity_j_right(obs_j_right.velocity.x(), obs_j_right.velocity.y());

                            auto *f_sc = new ProjectionTwoFrameTwoCamFactor(
                                pts_i, pts_j_right, velocity_i, velocity_j_right, td_i, td_j,
                                config_.focal_length, obs_j_left.has_cov, obs_j_left.sqrt_info_px);

                            auto *rbi_sc = new ResidualBlockInfo(
                                f_sc, loss_function,
                                std::vector<double *>{para_Pose_[0].data(), para_Pose_[imu_j].data(),
                                                      para_Ex_Pose_[0].data(), para_Ex_Pose_[1].data(),
                                                      para_inv_depth, para_Td_.data()},
                                std::vector<int>{0, 4});

                            marginalization_info->addResidualBlockInfo(rbi_sc);
                        }
                    }
                }
            }

            auto marg_start = std::chrono::high_resolution_clock::now();
            marginalization_info->preMarginalize();
            const int marg_threads = (config_.num_threads > 0)
                                         ? config_.num_threads
                                         : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 2);
            marginalization_info->marginalize(marg_threads);
            auto marg_end = std::chrono::high_resolution_clock::now();
            double marg_ms = std::chrono::duration<double, std::milli>(marg_end - marg_start).count();
            RCLCPP_DEBUG(rclcpp::get_logger("ceres_backend"), "Marginalize: %.1fms", marg_ms);

            std::unordered_map<std::uintptr_t, double *> addr_shift;
            for (int i = 1; i <= WINDOW_SIZE; i++)
            {
                addr_shift[reinterpret_cast<std::uintptr_t>(para_Pose_[i].data())] = para_Pose_[i - 1].data();
                addr_shift[reinterpret_cast<std::uintptr_t>(para_SpeedBias_[i].data())] = para_SpeedBias_[i - 1].data();
            }
            addr_shift[reinterpret_cast<std::uintptr_t>(para_Ex_Pose_[0].data())] = para_Ex_Pose_[0].data();
            addr_shift[reinterpret_cast<std::uintptr_t>(para_Ex_Pose_[1].data())] = para_Ex_Pose_[1].data();
            addr_shift[reinterpret_cast<std::uintptr_t>(para_Td_.data())] = para_Td_.data();

            auto *raw_info = marginalization_info.get();
            last_marginalization_parameter_blocks_ = raw_info->getParameterBlocks(addr_shift);
            last_marginalization_info_ = std::move(marginalization_info);

            std::vector<int> features_to_remove;
            if (feature_manager)
            {
                const auto &features = feature_manager->getFeatures();
                for (const auto &[id, arr] : para_Feature_)
                {
                    auto it = features.find(id);
                    if (it != features.end() && it->second.start_frame == 0)
                        features_to_remove.push_back(id);
                }
            }
            for (int id : features_to_remove)
                para_Feature_.erase(id);
        }

        void CeresBackend::marginalizeNewFrame(int frame_count,
                                               Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs,
                                               Eigen::Vector3d *Vs, Eigen::Vector3d *Bas, Eigen::Vector3d *Bgs,
                                               Eigen::Vector3d *tic, Eigen::Matrix3d *ric)
        {
            if (!last_marginalization_info_ ||
                std::count(last_marginalization_parameter_blocks_.begin(),
                           last_marginalization_parameter_blocks_.end(),
                           para_Pose_[frame_count - 1].data()) == 0)
            {
                return;
            }

            auto marginalization_info = std::make_unique<MarginalizationInfo>();
            vector2double(frame_count, Ps, Rs, Vs, Bas, Bgs, tic, ric);

            if (last_marginalization_info_ && last_marginalization_info_->valid)
            {
                std::vector<int> drop_set;
                for (size_t i = 0; i < last_marginalization_parameter_blocks_.size(); i++)
                {
                    if (last_marginalization_parameter_blocks_[i] == para_Pose_[frame_count - 1].data())
                        drop_set.push_back(static_cast<int>(i));
                }

                auto *marginalization_factor =
                    new MarginalizationFactor(last_marginalization_info_.get());

                auto *residual_block_info = new ResidualBlockInfo(
                    marginalization_factor, nullptr,
                    last_marginalization_parameter_blocks_, drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            marginalization_info->preMarginalize();
            {
                const int marg_threads = (config_.num_threads > 0)
                                             ? config_.num_threads
                                             : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 2);
                marginalization_info->marginalize(marg_threads);
            }

            std::unordered_map<std::uintptr_t, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == frame_count - 1)
                    continue;
                else if (i == frame_count)
                {
                    addr_shift[reinterpret_cast<std::uintptr_t>(para_Pose_[i].data())] = para_Pose_[i - 1].data();
                    addr_shift[reinterpret_cast<std::uintptr_t>(para_SpeedBias_[i].data())] = para_SpeedBias_[i - 1].data();
                }
                else
                {
                    addr_shift[reinterpret_cast<std::uintptr_t>(para_Pose_[i].data())] = para_Pose_[i].data();
                    addr_shift[reinterpret_cast<std::uintptr_t>(para_SpeedBias_[i].data())] = para_SpeedBias_[i].data();
                }
            }
            addr_shift[reinterpret_cast<std::uintptr_t>(para_Ex_Pose_[0].data())] = para_Ex_Pose_[0].data();
            addr_shift[reinterpret_cast<std::uintptr_t>(para_Ex_Pose_[1].data())] = para_Ex_Pose_[1].data();
            addr_shift[reinterpret_cast<std::uintptr_t>(para_Td_.data())] = para_Td_.data();

            auto *raw_info = marginalization_info.get();
            last_marginalization_parameter_blocks_ = raw_info->getParameterBlocks(addr_shift);
            last_marginalization_info_ = std::move(marginalization_info);
        }

    } // namespace perception
} // namespace uosm
