/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include "../utility/math_util.hpp"
#include <algorithm>
#include <cmath>

namespace projection
{

    // Extract pose (position + quaternion) from a 7-element Ceres parameter block
    inline void extractPose(const double *param, Eigen::Vector3d &P, Eigen::Quaterniond &Q)
    {
        P = Eigen::Vector3d(param[0], param[1], param[2]);
        Q = Eigen::Quaterniond(param[6], param[3], param[4], param[5]);
    }

    // Time-delay compensation: shift observation by velocity * (td - td_orig)
    inline Eigen::Vector3d tdCompensate(const Eigen::Vector3d &pt,
                                        const Eigen::Vector3d &vel,
                                        double td, double td_orig)
    {
        return pt - (td - td_orig) * vel;
    }

    // Camera → IMU frame transform: R_ic * pt_cam + t_ic
    inline Eigen::Vector3d cam2imu(const Eigen::Vector3d &pt_cam,
                                   const Eigen::Quaterniond &qic,
                                   const Eigen::Vector3d &tic)
    {
        return qic * pt_cam + tic;
    }

    // IMU → world frame transform: Q_i * pt_imu + P_i
    inline Eigen::Vector3d imu2world(const Eigen::Vector3d &pt_imu,
                                     const Eigen::Quaterniond &Qi,
                                     const Eigen::Vector3d &Pi)
    {
        return Qi * pt_imu + Pi;
    }

    // World → IMU frame transform: Q_j^{-1} * (pt_w - P_j)
    inline Eigen::Vector3d world2imu(const Eigen::Vector3d &pt_w,
                                     const Eigen::Quaterniond &Qj,
                                     const Eigen::Vector3d &Pj)
    {
        return Qj.inverse() * (pt_w - Pj);
    }

    // IMU → camera frame transform: R_ic^{-1} * (pt_imu - t_ic)
    inline Eigen::Vector3d imu2cam(const Eigen::Vector3d &pt_imu,
                                   const Eigen::Quaterniond &qic,
                                   const Eigen::Vector3d &tic)
    {
        return qic.inverse() * (pt_imu - tic);
    }

    // 2x3 projection Jacobian (reduce matrix): d(proj)/d(pt_cam)
    inline Eigen::Matrix<double, 2, 3> projJacobian(const Eigen::Vector3d &pt_cam,
                                                    double dep,
                                                    const Eigen::Matrix2d &sqrt_info)
    {
        Eigen::Matrix<double, 2, 3> reduce;
        reduce << 1.0 / dep, 0, -pt_cam(0) / (dep * dep),
            0, 1.0 / dep, -pt_cam(1) / (dep * dep);
        return sqrt_info * reduce;
    }

    // Compute normalized projection residual
    inline Eigen::Vector2d projResidual(const Eigen::Vector3d &pt_cam,
                                        const Eigen::Vector3d &obs,
                                        double dep,
                                        const Eigen::Matrix2d &sqrt_info)
    {
        return sqrt_info * ((pt_cam / dep).head<2>() - obs.head<2>());
    }

} // namespace projection

// ==================== Factor: Two Frames, One Camera (mono) ====================
// Parameters: pose_i(7), pose_j(7), extrinsic(7), inv_depth(1), td(1)

class ProjectionTwoFrameOneCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
public:
    ProjectionTwoFrameOneCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
                                   const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                   double _td_i, double _td_j,
                                   double _focal_length = 460.0,
                                   bool _has_cov = false,
                                   const Eigen::Matrix2d &_sqrt_info_px = Eigen::Matrix2d::Identity())
        : pts_i(_pts_i), pts_j(_pts_j), td_i(_td_i), td_j(_td_j), focal_length(_focal_length)
    {
        velocity_i << _velocity_i.x(), _velocity_i.y(), 0.0;
        velocity_j << _velocity_j.x(), _velocity_j.y(), 0.0;
        // RaCo learned covariance (anisotropic) when available, else isotropic focal/1.5.
        sqrt_info = _has_cov ? _sqrt_info_px : (focal_length / 1.5) * Eigen::Matrix2d::Identity();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Vector3d Pi, Pj, tic;
        Eigen::Quaterniond Qi, Qj, qic;
        projection::extractPose(parameters[0], Pi, Qi);
        projection::extractPose(parameters[1], Pj, Qj);
        projection::extractPose(parameters[2], tic, qic);
        double inv_dep_i = parameters[3][0];
        double td = parameters[4][0];

        // Forward projection chain
        Eigen::Vector3d pts_i_td = projection::tdCompensate(pts_i, velocity_i, td, td_i);
        Eigen::Vector3d pts_j_td = projection::tdCompensate(pts_j, velocity_j, td, td_j);
        Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Vector3d pts_imu_i = projection::cam2imu(pts_camera_i, qic, tic);
        Eigen::Vector3d pts_w = projection::imu2world(pts_imu_i, Qi, Pi);
        Eigen::Vector3d pts_imu_j = projection::world2imu(pts_w, Qj, Pj);
        Eigen::Vector3d pts_camera_j = projection::imu2cam(pts_imu_j, qic, tic);

        double dep_j = pts_camera_j.z();
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = projection::projResidual(pts_camera_j, pts_j_td, dep_j, sqrt_info);

        if (jacobians)
        {
            Eigen::Matrix3d Ri = Qi.toRotationMatrix();
            Eigen::Matrix3d Rj = Qj.toRotationMatrix();
            Eigen::Matrix3d ric = qic.toRotationMatrix();
            auto reduce = projection::projJacobian(pts_camera_j, dep_j, sqrt_info);

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = ric.transpose() * Rj.transpose();
                jaco.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[1]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = ric.transpose() * -Rj.transpose();
                jaco.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[2]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
                Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
                jaco.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) +
                                      Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                      Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[3])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[3]);
                J = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
            }

            if (jacobians[4])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[4]);
                J = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0 +
                    sqrt_info * velocity_j.head<2>();
            }
        }

        return true;
    }

private:
    Eigen::Vector3d pts_i, pts_j;
    Eigen::Vector3d velocity_i, velocity_j;
    double td_i, td_j;
    double focal_length;
    Eigen::Matrix2d sqrt_info;
};

// ==================== Factor: Two Frames, Two Cameras (stereo cross-cam) ====================
// Parameters: pose_i(7), pose_j(7), extrinsic_left(7), extrinsic_right(7), inv_depth(1), td(1)

class ProjectionTwoFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1, 1>
{
public:
    ProjectionTwoFrameTwoCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
                                   const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                   double _td_i, double _td_j,
                                   double _focal_length = 460.0,
                                   bool _has_cov = false,
                                   const Eigen::Matrix2d &_sqrt_info_px = Eigen::Matrix2d::Identity())
        : pts_i(_pts_i), pts_j(_pts_j), td_i(_td_i), td_j(_td_j), focal_length(_focal_length)
    {
        velocity_i << _velocity_i.x(), _velocity_i.y(), 0.0;
        velocity_j << _velocity_j.x(), _velocity_j.y(), 0.0;
        // RaCo learned covariance (anisotropic) when available, else isotropic focal/1.5.
        sqrt_info = _has_cov ? _sqrt_info_px : (focal_length / 1.5) * Eigen::Matrix2d::Identity();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Vector3d Pi, Pj, tic, tic2;
        Eigen::Quaterniond Qi, Qj, qic, qic2;
        projection::extractPose(parameters[0], Pi, Qi);
        projection::extractPose(parameters[1], Pj, Qj);
        projection::extractPose(parameters[2], tic, qic);
        projection::extractPose(parameters[3], tic2, qic2);
        double inv_dep_i = parameters[4][0];
        double td = parameters[5][0];

        // Forward projection: left cam i → world → right cam j
        Eigen::Vector3d pts_i_td = projection::tdCompensate(pts_i, velocity_i, td, td_i);
        Eigen::Vector3d pts_j_td = projection::tdCompensate(pts_j, velocity_j, td, td_j);
        Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Vector3d pts_imu_i = projection::cam2imu(pts_camera_i, qic, tic);
        Eigen::Vector3d pts_w = projection::imu2world(pts_imu_i, Qi, Pi);
        Eigen::Vector3d pts_imu_j = projection::world2imu(pts_w, Qj, Pj);
        Eigen::Vector3d pts_camera_j = projection::imu2cam(pts_imu_j, qic2, tic2);

        double dep_j = pts_camera_j.z();
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = projection::projResidual(pts_camera_j, pts_j_td, dep_j, sqrt_info);

        if (jacobians)
        {
            Eigen::Matrix3d Ri = Qi.toRotationMatrix();
            Eigen::Matrix3d Rj = Qj.toRotationMatrix();
            Eigen::Matrix3d ric = qic.toRotationMatrix();
            Eigen::Matrix3d ric2 = qic2.toRotationMatrix();
            auto reduce = projection::projJacobian(pts_camera_j, dep_j, sqrt_info);

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = ric2.transpose() * Rj.transpose();
                jaco.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[1]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = ric2.transpose() * -Rj.transpose();
                jaco.rightCols<3>() = ric2.transpose() * Utility::skewSymmetric(pts_imu_j);
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[2]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = ric2.transpose() * Rj.transpose() * Ri;
                jaco.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * ric * -Utility::skewSymmetric(pts_camera_i);
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[3]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = -ric2.transpose();
                jaco.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[4])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[4]);
                J = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
            }

            if (jacobians[5])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[5]);
                J = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0 +
                    sqrt_info * velocity_j.head<2>();
            }
        }

        return true;
    }

private:
    Eigen::Vector3d pts_i, pts_j;
    Eigen::Vector3d velocity_i, velocity_j;
    double td_i, td_j;
    double focal_length;
    Eigen::Matrix2d sqrt_info;
};

// ==================== Factor: One Frame, Two Cameras (stereo baseline) ====================
// Parameters: extrinsic_left(7), extrinsic_right(7), inv_depth(1), td(1)

class ProjectionOneFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 1, 1>
{
public:
    ProjectionOneFrameTwoCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
                                   const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                   double _td_i, double _td_j,
                                   double _focal_length = 460.0)
        : pts_i(_pts_i), pts_j(_pts_j), td_i(_td_i), td_j(_td_j), focal_length(_focal_length)
    {
        velocity_i << _velocity_i.x(), _velocity_i.y(), 0.0;
        velocity_j << _velocity_j.x(), _velocity_j.y(), 0.0;
        sqrt_info = focal_length / 1.5 * Eigen::Matrix2d::Identity();
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Vector3d tic, tic2;
        Eigen::Quaterniond qic, qic2;
        projection::extractPose(parameters[0], tic, qic);
        projection::extractPose(parameters[1], tic2, qic2);
        double inv_dep_i = parameters[2][0];
        double td = parameters[3][0];

        // Forward projection: left cam → IMU → right cam (same frame)
        Eigen::Vector3d pts_i_td = projection::tdCompensate(pts_i, velocity_i, td, td_i);
        Eigen::Vector3d pts_j_td = projection::tdCompensate(pts_j, velocity_j, td, td_j);
        Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Vector3d pts_imu = projection::cam2imu(pts_camera_i, qic, tic);
        Eigen::Vector3d pts_camera_j = projection::imu2cam(pts_imu, qic2, tic2);

        double dep_j = pts_camera_j.z();
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = projection::projResidual(pts_camera_j, pts_j_td, dep_j, sqrt_info);

        if (jacobians)
        {
            Eigen::Matrix3d ric = qic.toRotationMatrix();
            Eigen::Matrix3d ric2 = qic2.toRotationMatrix();
            auto reduce = projection::projJacobian(pts_camera_j, dep_j, sqrt_info);

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = ric2.transpose();
                jaco.rightCols<3>() = ric2.transpose() * ric * -Utility::skewSymmetric(pts_camera_i);
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[1]);
                Eigen::Matrix<double, 3, 6> jaco;
                jaco.leftCols<3>() = -ric2.transpose();
                jaco.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);
                J.leftCols<6>() = reduce * jaco;
                J.rightCols<1>().setZero();
            }

            if (jacobians[2])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[2]);
                J = reduce * ric2.transpose() * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
            }

            if (jacobians[3])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[3]);
                J = reduce * ric2.transpose() * ric * velocity_i / inv_dep_i * -1.0 +
                    sqrt_info * velocity_j.head<2>();
            }
        }

        return true;
    }

private:
    Eigen::Vector3d pts_i, pts_j;
    Eigen::Vector3d velocity_i, velocity_j;
    double td_i, td_j;
    double focal_length;
    Eigen::Matrix2d sqrt_info;
};
