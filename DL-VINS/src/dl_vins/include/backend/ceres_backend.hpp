#pragma once

#include "backend_interface.hpp"
#include <ceres/ceres.h>
#include "imu_factor.hpp"
#include "reprojection_factor.hpp"
#include "marginalization_factor.hpp"
#include "pose_manifold.hpp"

#include <array>
#include <memory>
#include <unordered_map>

namespace uosm
{
    namespace perception
    {
        class CeresBackend : public IBackendSolver
        {
        public:
            explicit CeresBackend(const BackendSolverConfig &config = {});
            ~CeresBackend() override = default;

            bool solve(
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
                int marginalization_flag) override;

            void marginalize(
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
                bool marginalize_old) override;

            SolverStatistics getStatistics() const override { return stats_; }
            void reset() override;

        private:
            BackendSolverConfig config_;
            SolverStatistics stats_;

            std::unique_ptr<MarginalizationInfo> last_marginalization_info_;
            std::vector<double *> last_marginalization_parameter_blocks_;
            bool open_ex_estimation_ = false;
            std::array<Eigen::Vector3d, 2> ex_ref_t_{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
            std::array<Eigen::Quaterniond, 2> ex_ref_q_{Eigen::Quaterniond::Identity(), Eigen::Quaterniond::Identity()};

            std::vector<std::array<double, 7>> para_Pose_;
            std::vector<std::array<double, 9>> para_SpeedBias_;
            std::array<std::array<double, 7>, 2> para_Ex_Pose_{};
            std::array<double, 1> para_Td_{};
            std::unordered_map<int, std::array<double, 1>> para_Feature_;

            void initParameters();

            void vector2double(int frame_count,
                               Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs,
                               Eigen::Vector3d *Vs, Eigen::Vector3d *Bas, Eigen::Vector3d *Bgs,
                               Eigen::Vector3d *tic, Eigen::Matrix3d *ric);

            void double2vector(int frame_count,
                               Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs,
                               Eigen::Vector3d *Vs, Eigen::Vector3d *Bas, Eigen::Vector3d *Bgs,
                               Eigen::Vector3d *tic, Eigen::Matrix3d *ric);

            void addIMUResidualBlocks(ceres::Problem &problem, int frame_count,
                                      std::vector<std::unique_ptr<IntegrationBase>> &pre_integrations,
                                      std::vector<ceres::ResidualBlockId> *imu_ids = nullptr);

            // Per-type residual-block IDs for cost attribution (any may be null).
            struct ReprojBlockIds
            {
                std::vector<ceres::ResidualBlockId> temporal;  // ProjectionTwoFrameOneCamFactor
                std::vector<ceres::ResidualBlockId> stereo;    // ProjectionOneFrameTwoCamFactor
                std::vector<ceres::ResidualBlockId> cross_cam; // ProjectionTwoFrameTwoCamFactor
            };
            void addVisualResidualBlocks(ceres::Problem &problem, int frame_count,
                                         std::shared_ptr<FeatureManager> feature_manager,
                                         ReprojBlockIds *reproj_ids = nullptr);

            void addMarginalizationResidualBlock(ceres::Problem &problem,
                                                 std::vector<ceres::ResidualBlockId> *marg_ids = nullptr);

            void marginalizeOldFrame(int frame_count,
                                     Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs,
                                     Eigen::Vector3d *Vs, Eigen::Vector3d *Bas, Eigen::Vector3d *Bgs,
                                     Eigen::Vector3d *tic, Eigen::Matrix3d *ric,
                                     std::vector<std::unique_ptr<IntegrationBase>> &pre_integrations,
                                     std::shared_ptr<FeatureManager> feature_manager);

            void marginalizeNewFrame(int frame_count,
                                     Eigen::Vector3d *Ps, Eigen::Matrix3d *Rs,
                                     Eigen::Vector3d *Vs, Eigen::Vector3d *Bas, Eigen::Vector3d *Bgs,
                                     Eigen::Vector3d *tic, Eigen::Matrix3d *ric);
        };

    } // namespace perception
} // namespace uosm
