/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "../../include/initial/initial_alignment.hpp"
#include <iostream>
#include <rclcpp/logging.hpp>

void VisualIMUAlignment::solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame,
                                            Eigen::Vector3d *Bgs,
                                            size_t window_size)
{
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    int constraint_count = 0;

    auto frame_i = all_image_frame.begin();
    auto frame_j = std::next(frame_i);

    while (frame_j != all_image_frame.end())
    {
        if (frame_j->second.pre_integration == nullptr)
        {
            frame_i = frame_j;
            ++frame_j;
            continue;
        }

        // Visual rotation from i to j
        const Eigen::Quaterniond q_visual(frame_i->second.R.transpose() * frame_j->second.R);

        // IMU preintegrated rotation and Jacobian
        const Eigen::Quaterniond &q_imu = frame_j->second.pre_integration->delta_q;
        const Eigen::Matrix3d &J_bg = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

        // Small angle approximation: error in preintegration frame
        const Eigen::Vector3d residual = 2.0 * (q_imu.inverse() * q_visual).vec();

        A += J_bg.transpose() * J_bg;
        b += J_bg.transpose() * residual;
        ++constraint_count;

        frame_i = frame_j;
        ++frame_j;
    }

    if (constraint_count < 2)
    {
        RCLCPP_WARN(rclcpp::get_logger("estimator"), "Gyro Bias: Not enough constraints: %d", constraint_count);
        return;
    }

    const Eigen::Vector3d delta_bg = A.ldlt().solve(b);

    RCLCPP_DEBUG(rclcpp::get_logger("estimator"),
                 "Gyro Bias Correction: [%.6f, %.6f, %.6f] (%d constraints)",
                 delta_bg.x(), delta_bg.y(), delta_bg.z(), constraint_count);

    // Apply to window frames
    for (size_t i = 0; i <= window_size; ++i)
    {
        Bgs[i] += delta_bg;
    }

    // Repropagate preintegrations
    for (auto &[timestamp, frame] : all_image_frame)
    {
        if (frame.pre_integration != nullptr)
        {
            frame.pre_integration->repropagate(Eigen::Vector3d::Zero(), Bgs[0]);
        }
    }
}

Eigen::MatrixXd VisualIMUAlignment::tangentBasis(Eigen::Vector3d &g0)
{
    Eigen::Vector3d a = g0.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    Eigen::Vector3d b = (tmp - a * (a.transpose() * tmp)).normalized();
    Eigen::Vector3d c = a.cross(b);
    Eigen::MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

bool VisualIMUAlignment::linearAlignment(std::map<double, ImageFrame> &all_image_frame,
                                         Eigen::Vector3d &g,
                                         Eigen::VectorXd &x,
                                         double gravity_norm,
                                         const Eigen::Vector3d &tic0,
                                         double g_error_tolerance)
{
    int all_frame_count = static_cast<int>(all_image_frame.size());
    int n_state = all_frame_count * 3 + 3 + 1;

    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    auto frame_i = all_image_frame.begin();
    auto frame_j = std::next(frame_i);
    int i = 0;
    for (; frame_j != all_image_frame.end(); frame_i = frame_j, ++frame_j, ++i)
    {
        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        // Position constraints
        tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p +
                                  frame_i->second.R.transpose() * frame_j->second.R * tic0 - tic0;

        // Velocity constraints
        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

        Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Identity();

        Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }

    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);

    double s = x(n_state - 1) / 100.0;
    g = x.segment<3>(n_state - 4);

    RCLCPP_DEBUG(rclcpp::get_logger("estimator"),
                 "LinearAlignment: g=[%.4f, %.4f, %.4f] |g|=%.4f, scale=%.4f",
                 g.x(), g.y(), g.z(), g.norm(), s);

    if (std::fabs(g.norm() - gravity_norm) > g_error_tolerance || s < 0)
    {
        RCLCPP_WARN(rclcpp::get_logger("estimator"),
                    "LinearAlignment failed: |g|=%.4f (expected ~%.4f, tol=%.3f), scale=%.4f",
                    g.norm(), gravity_norm, g_error_tolerance, s);
        return false;
    }

    refineGravity(all_image_frame, g, x, gravity_norm, tic0);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;

    RCLCPP_DEBUG(rclcpp::get_logger("estimator"),
                 "RefineGravity: g=[%.4f, %.4f, %.4f] |g|=%.4f, scale=%.4f",
                 g.x(), g.y(), g.z(), g.norm(), s);

    if (s < 0.0)
        return false;

    return true;
}

void VisualIMUAlignment::refineGravity(std::map<double, ImageFrame> &all_image_frame,
                                       Eigen::Vector3d &g,
                                       Eigen::VectorXd &x,
                                       double gravity_norm,
                                       const Eigen::Vector3d &tic0)
{
    Eigen::Vector3d g0 = g.normalized() * gravity_norm;

    int all_frame_count = static_cast<int>(all_image_frame.size());
    int n_state = all_frame_count * 3 + 2 + 1;

    Eigen::MatrixXd A{n_state, n_state};
    Eigen::VectorXd b{n_state};

    for (int k = 0; k < 4; k++)
    {
        Eigen::MatrixXd lxly = tangentBasis(g0);

        A.setZero();
        b.setZero();

        auto frame_i = all_image_frame.begin();
        auto frame_j = std::next(frame_i);
        int i = 0;
        for (; frame_j != all_image_frame.end(); frame_i = frame_j, ++frame_j, ++i)
        {
            Eigen::MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            Eigen::VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            // Position constraints
            tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p +
                                      frame_i->second.R.transpose() * frame_j->second.R * tic0 - tic0 -
                                      frame_i->second.R.transpose() * dt * dt / 2 * g0;

            // Velocity constraints
            tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v -
                                      frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * g0;

            Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Identity();

            Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }

        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        Eigen::VectorXd dg = x.segment<2>(n_state - 3);
        g0 = (g0 + lxly * dg).normalized() * gravity_norm;
    }

    g = g0;
}

bool VisualIMUAlignment::solveVelocityGravityScale(
    std::map<double, ImageFrame> &all_image_frame,
    Eigen::Vector3d *Bgs,
    Eigen::Vector3d &g,
    Eigen::VectorXd &x,
    double gravity_norm,
    const Eigen::Vector3d &tic0,
    size_t window_size,
    double g_error_tolerance)
{
    solveGyroscopeBias(all_image_frame, Bgs, window_size);

    if (linearAlignment(all_image_frame, g, x, gravity_norm, tic0, g_error_tolerance))
        return true;

    return false;
}
