/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <map>
#include <memory>
#include <eigen3/Eigen/Dense>
#include "../imu/integration_base.hpp"

struct ImageFrame
{
    ImageFrame() = default;
    explicit ImageFrame(double timestamp) : t(timestamp) {}

    double t{0.0};
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d T{Eigen::Vector3d::Zero()};
    std::shared_ptr<IntegrationBase> pre_integration;
    bool is_key_frame{false};
};

class VisualIMUAlignment
{
public:
    /**
     * @brief Solve gyroscope bias using rotation constraints
     */
    static void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame,
                                   Eigen::Vector3d *Bgs,
                                   size_t window_size);

    /**
     * @brief Full visual-IMU alignment: gyro bias + velocity + gravity + scale
     * @param all_image_frame  Map of timestamp → ImageFrame with visual poses and pre-integration
     * @param Bgs              Gyroscope biases (updated in-place)
     * @param g                [out] Refined gravity vector in visual frame
     * @param x                [out] Solution vector: [v_0..v_N, gravity_or_perturbation, scale]
     * @param gravity_norm     Expected gravity magnitude (9.81)
     * @param tic0             Camera-to-IMU translation for left camera
     * @param window_size      Sliding window size
     * @return true if alignment succeeded (valid gravity magnitude and positive scale)
     */
    static bool solveVelocityGravityScale(
        std::map<double, ImageFrame> &all_image_frame,
        Eigen::Vector3d *Bgs,
        Eigen::Vector3d &g,
        Eigen::VectorXd &x,
        double gravity_norm,
        const Eigen::Vector3d &tic0,
        size_t window_size,
        double g_error_tolerance = 0.5f);

private:
    /**
     * @brief Linear solve for per-frame velocities, 3D gravity, and scale
     */
    static bool linearAlignment(std::map<double, ImageFrame> &all_image_frame,
                                Eigen::Vector3d &g,
                                Eigen::VectorXd &x,
                                double gravity_norm,
                                const Eigen::Vector3d &tic0,
                                double g_error_tolerance);

    /**
     * @brief Iterative gravity refinement on 2-DOF manifold
     */
    static void refineGravity(std::map<double, ImageFrame> &all_image_frame,
                              Eigen::Vector3d &g,
                              Eigen::VectorXd &x,
                              double gravity_norm,
                              const Eigen::Vector3d &tic0);

    /**
     * @brief Compute orthonormal tangent basis perpendicular to gravity direction
     */
    static Eigen::MatrixXd tangentBasis(Eigen::Vector3d &g0);
};
