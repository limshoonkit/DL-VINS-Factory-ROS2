#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <map>
#include <memory>
#include <string>

#include "../imu/integration_base.hpp"
#include "../feature/feature_manager.hpp"

namespace uosm
{
    namespace perception
    {
        enum class BackendType
        {
            CERES = 0,    // Ceres NLLS
            HYPERION = 1, // TODO: Hyperion GBP
        };

        struct BackendSolverConfig
        {
            // Common settings
            int max_iterations = 8U;
            double focal_length = 460.0f;
            bool estimate_extrinsic = false;
            double initial_td = 0.0f;
            bool estimate_td = false;
            int min_observations = 4U;
            double max_imu_sum_dt = 10.0f;

            // Solver settings
            int num_threads = 0U;           // 0 = auto-detect from hardware_concurrency
            double max_solver_time = 0.04f; // Max solver wall time in seconds (real-time budget)
            bool use_gpu = false;           // Enable CUDA acceleration for DENSE_SCHUR
        };

        struct SolverStatistics
        {
            double final_cost = 0.0;
            int iterations = 0;
            bool converged = false;
            double solve_time_ms = 0.0;
            double marginalization_time_ms = 0.0;
            std::string solver_info;

            // Post-solve cost contribution by factor group
            double cost_marg = 0.0;
            double cost_imu = 0.0;
            double cost_temporal = 0.0;  // ProjectionTwoFrameOneCamFactor (left cam, two frames)
            double cost_stereo = 0.0;    // ProjectionOneFrameTwoCamFactor (left/right pair, one frame)
            double cost_cross_cam = 0.0; // ProjectionTwoFrameTwoCamFactor (left@i vs right@j)
        };

        class IBackendSolver
        {
        public:
            virtual ~IBackendSolver() = default;

            /**
             * @brief Solve the sliding window optimization
             *
             * @param frame_count Current frame count in window
             * @param Ps Positions for each frame [in/out]
             * @param Rs Rotations for each frame [in/out]
             * @param Vs Velocities for each frame [in/out]
             * @param Bas Accelerometer biases [in/out]
             * @param Bgs Gyroscope biases [in/out]
             * @param tic Camera-IMU translation [in/out]
             * @param ric Camera-IMU rotation [in/out]
             * @param pre_integrations IMU preintegrations between frames
             * @param feature_manager Feature manager with 3D points
             * @param marginalization_flag Indicates which frame to marginalize (0=old, 1=new)
             * @return true if optimization succeeded
             */
            virtual bool solve(
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
                int marginalization_flag) = 0;

            /**
             * @brief Perform marginalization after optimization
             * @param marginalize_old true = marginalize oldest frame, false = marginalize second newest
             */
            virtual void marginalize(
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
                bool marginalize_old) = 0;

            virtual SolverStatistics getStatistics() const = 0;
            virtual void reset() = 0;
        };

        /**
         * @brief Factory to create backend solvers
         */
        class BackendSolverFactory
        {
        public:
            static std::unique_ptr<IBackendSolver> create(
                BackendType type,
                const BackendSolverConfig &config = {});
        };

    } // namespace perception
} // namespace uosm