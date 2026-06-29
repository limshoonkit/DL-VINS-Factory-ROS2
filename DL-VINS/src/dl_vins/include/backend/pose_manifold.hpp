/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ceres/manifold.h>
#include <eigen3/Eigen/Dense>
#include "../utility/math_util.hpp"

class PoseManifold : public ceres::Manifold
{
public:
    int AmbientSize() const override { return 7; } // 3 for translation + 4 for quaternion
    int TangentSize() const override { return 6; } // 3 for translation + 3 for rotation

    bool Plus(const double *x, const double *delta,
              double *x_plus_delta) const override
    {
        // Translation part
        Eigen::Map<const Eigen::Vector3d> p(x);
        Eigen::Map<const Eigen::Vector3d> dp(delta);
        Eigen::Map<Eigen::Vector3d> p_plus(x_plus_delta);
        p_plus = p + dp;

        // Rotation part
        Eigen::Map<const Eigen::Quaterniond> q(x + 3);
        Eigen::Map<const Eigen::Vector3d> dtheta(delta + 3);
        Eigen::Quaterniond dq = Utility::deltaQ(dtheta);
        Eigen::Map<Eigen::Quaterniond> q_plus(x_plus_delta + 3);
        q_plus = (q * dq).normalized();

        return true;
    }

    bool Minus(const double *y, const double *x,
               double *y_minus_x) const override
    {
        // Translation part
        Eigen::Map<const Eigen::Vector3d> p_y(y);
        Eigen::Map<const Eigen::Vector3d> p_x(x);
        Eigen::Map<Eigen::Vector3d> dp(y_minus_x);
        dp = p_y - p_x;

        // Rotation part
        Eigen::Map<const Eigen::Quaterniond> q_y(y + 3);
        Eigen::Map<const Eigen::Quaterniond> q_x(x + 3);
        Eigen::Map<Eigen::Vector3d> dtheta(y_minus_x + 3);

        // Compute the relative rotation
        Eigen::Quaterniond dq = q_x.conjugate() * q_y;

        // Convert to axis-angle, handling the special cases
        double w = dq.w();
        Eigen::Vector3d vec = dq.vec();

        if (w >= 1.0)
        {
            // Identity rotation
            dtheta.setZero();
        }
        else
        {
            double theta = 2.0 * std::acos(std::min(1.0, std::max(-1.0, w)));
            double sin_half_theta = std::sin(theta / 2.0);

            if (std::abs(sin_half_theta) < 1e-10)
            {
                // Small angle approximation
                dtheta = 2.0 * vec;
            }
            else
            {
                dtheta = theta / sin_half_theta * vec;
            }
        }

        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override
    {
        // The jacobian is a 7x6 matrix
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);

        // Set the top 6x6 block to identity
        j.topLeftCorner<3, 3>().setIdentity();
        j.topRightCorner<3, 3>().setZero();
        j.block<4, 3>(3, 0).setZero();

        // Simplified rotation jacobian at identity
        Eigen::Map<const Eigen::Quaterniond> q(x + 3);
        j.block<3, 3>(3, 3).setIdentity();
        j.bottomRows<1>().setZero(); // quaternion w component

        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override
    {
        // For SE(3) manifold, the minus jacobian at identity is also identity
        return PlusJacobian(x, jacobian);
    }
};