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
#include <iostream>
#include "../imu/integration_base.hpp"
#include "../utility/math_util.hpp"

// 15 — residual dimension (the IMU preintegration error vector):
// 3: position residual
// 3: orientation residual (RPY)
// 3: velocity residual
// 3: accelerometer bias residual
// 3: gyroscope bias residual

// 7, 9, 7, 9 — sizes of the four parameter blocks (states at time i and time j):
// 7 — Pose_i: position (3) + orientation quaternion (4)
// 9 — SpeedBias_i: velocity (3) + accel bias (3) + gyro bias (3)
// 7 — Pose_j: position (3) + orientation quaternion (4)
// 9 — SpeedBias_j: velocity (3) + accel bias (3) + gyro bias (3)
class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
public:
    IMUFactor() = delete;

    IMUFactor(IntegrationBase *_pre_integration)
        : pre_integration(_pre_integration)
    {
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        // Extract state i
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        // Extract state j
        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

        // Compute residual using preintegration
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

        // Compute square root information matrix for weighting
        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
                                                      pre_integration->covariance.inverse())
                                                      .matrixL()
                                                      .transpose();

        residual = sqrt_info * residual;

        if (jacobians)
        {
            const double sum_dt = pre_integration->sum_dt;

            // Get preintegration Jacobians
            const Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
            const Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);
            const Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
            const Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
            const Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

            // Bias correction
            const Eigen::Vector3d dbg = Bgi - pre_integration->linearized_bg;

            const Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q *
                                                         Utility::deltaQ(dq_dbg * dbg);

            if (jacobians[0]) // Jacobian w.r.t. pose i
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                // d_residual / d_Pi
                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();

                // d_residual / d_Qi (for rotation part)
                jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(
                    Qi.inverse() * (0.5 * sum_dt * sum_dt * pre_integration->gravity +
                                    Pj - Pi - sum_dt * Vi));

                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) *
                                                          Utility::Qright(corrected_delta_q))
                                                             .bottomRightCorner<3, 3>();

                jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(
                    Qi.inverse() * (sum_dt * pre_integration->gravity + Vj - Vi));

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    std::cerr << "[IMUFactor] Numerical instability in jacobian_pose_i, resetting\n";
                    jacobian_pose_i.setZero();
                }
            }

            if (jacobians[1]) // Jacobian w.r.t. velocity and bias i
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();

                // d_residual / d_Vi
                jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();

                // d_residual / d_Bai
                jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

                // d_residual / d_Bgi
                jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(
                                                                         Qj.inverse() * Qi * pre_integration->delta_q)
                                                                         .bottomRightCorner<3, 3>() *
                                                                    dq_dbg;
                jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;
                jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                if (jacobian_speedbias_i.maxCoeff() > 1e8 || jacobian_speedbias_i.minCoeff() < -1e8)
                {
                    std::cerr << "[IMUFactor] Numerical instability in jacobian_speedbias_i, resetting\n";
                    jacobian_speedbias_i.setZero();
                }
            }

            if (jacobians[2]) // Jacobian w.r.t. pose j
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();

                // d_residual / d_Pj
                jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

                // d_residual / d_Qj
                jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(
                                                            corrected_delta_q.inverse() * Qi.inverse() * Qj)
                                                            .bottomRightCorner<3, 3>();

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                if (jacobian_pose_j.maxCoeff() > 1e8 || jacobian_pose_j.minCoeff() < -1e8)
                {
                    std::cerr << "[IMUFactor] Numerical instability in jacobian_pose_j, resetting\n";
                    jacobian_pose_j.setZero();
                }
            }

            if (jacobians[3]) // Jacobian w.r.t. velocity and bias j
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();

                // d_residual / d_Vj
                jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

                // d_residual / d_Baj
                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

                // d_residual / d_Bgj
                jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

                if (jacobian_speedbias_j.maxCoeff() > 1e8 || jacobian_speedbias_j.minCoeff() < -1e8)
                {
                    std::cerr << "[IMUFactor] Numerical instability in jacobian_speedbias_j, resetting\n";
                    jacobian_speedbias_j.setZero();
                }
            }
        }

        return true;
    }

private:
    IntegrationBase *pre_integration;
};
