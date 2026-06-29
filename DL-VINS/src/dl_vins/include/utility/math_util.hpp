/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <cmath>
#include <cassert>
#include <cstring>
#include <eigen3/Eigen/Dense>

class Utility
{
public:
    template <typename Derived>
    static inline Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        // Avoid unnecessary copy by using Map directly
        const Scalar_t norm = theta.norm() / static_cast<Scalar_t>(2.0);

        if (norm < static_cast<Scalar_t>(1e-5))
        {
            // For very small rotations, use a more accurate approximation
            return Eigen::Quaternion<Scalar_t>(
                       static_cast<Scalar_t>(1.0),
                       theta.x() / static_cast<Scalar_t>(2.0),
                       theta.y() / static_cast<Scalar_t>(2.0),
                       theta.z() / static_cast<Scalar_t>(2.0))
                .normalized();
        }

        const Scalar_t sine = std::sin(norm);
        return Eigen::Quaternion<Scalar_t>(
            std::cos(norm),
            theta.x() * sine / (static_cast<Scalar_t>(2.0) * norm),
            theta.y() * sine / (static_cast<Scalar_t>(2.0) * norm),
            theta.z() * sine / (static_cast<Scalar_t>(2.0) * norm));
    }

    template <typename Derived>
    static inline Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1,
                      "skewSymmetric requires a 3x1 vector");

        using Scalar_t = typename Derived::Scalar;
        Eigen::Matrix<Scalar_t, 3, 3> ans;
        ans << 0, -q(2), q(1),
            q(2), 0, -q(0),
            -q(1), q(0), 0;
        return ans;
    }

    // Fast skew symmetric for double vectors (most common case)
    static inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &q)
    {
        Eigen::Matrix3d ans;
        ans << 0, -q(2), q(1),
            q(2), 0, -q(0),
            -q(1), q(0), 0;
        return ans;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        // printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
        // Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
        // printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
        // return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
        return q;
    }

    template <typename Derived>
    static inline Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
    {
        using Scalar_t = typename Derived::Scalar;
        const auto &qq = positify(q);

        Eigen::Matrix<Scalar_t, 4, 4> ans;
        ans(0, 0) = qq.w();
        ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
        ans.template block<3, 1>(1, 0) = qq.vec();

        // Use noalias() to avoid temporary
        ans.template block<3, 3>(1, 1).noalias() =
            qq.w() * Eigen::Matrix<Scalar_t, 3, 3>::Identity() + skewSymmetric(qq.vec());

        return ans;
    }

    template <typename Derived>
    static inline Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
    {
        using Scalar_t = typename Derived::Scalar;
        const auto &pp = positify(p);

        Eigen::Matrix<Scalar_t, 4, 4> ans;
        ans(0, 0) = pp.w();
        ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
        ans.template block<3, 1>(1, 0) = pp.vec();

        // Use noalias() to avoid temporary
        ans.template block<3, 3>(1, 1).noalias() =
            pp.w() * Eigen::Matrix<Scalar_t, 3, 3>::Identity() - skewSymmetric(pp.vec());

        return ans;
    }

    static inline Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        // Cache these values to avoid repeated access
        const double &n1 = R(0, 0);
        const double &n2 = R(1, 0);
        const double &n3 = R(2, 0);

        const double y = std::atan2(n2, n1);
        const double cos_y = std::cos(y);
        const double sin_y = std::sin(y);

        const double p = std::atan2(-n3, n1 * cos_y + n2 * sin_y);

        const double &a1 = R(0, 2);
        const double &a2 = R(1, 2);
        const double &o1 = R(0, 1);
        const double &o2 = R(1, 1);

        const double r = std::atan2(a1 * sin_y - a2 * cos_y, -o1 * sin_y + o2 * cos_y);

        static constexpr double RAD_TO_DEG = 180.0 / M_PI;
        return Eigen::Vector3d(y, p, r) * RAD_TO_DEG;
    }

    template <typename Derived>
    static inline Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        using Scalar_t = typename Derived::Scalar;
        static constexpr Scalar_t DEG_TO_RAD = M_PI / 180.0;

        const Scalar_t y = ypr(0) * DEG_TO_RAD;
        const Scalar_t p = ypr(1) * DEG_TO_RAD;
        const Scalar_t r = ypr(2) * DEG_TO_RAD;

        const Scalar_t cy = std::cos(y);
        const Scalar_t sy = std::sin(y);
        const Scalar_t cp = std::cos(p);
        const Scalar_t sp = std::sin(p);
        const Scalar_t cr = std::cos(r);
        const Scalar_t sr = std::sin(r);

        Eigen::Matrix<Scalar_t, 3, 3> R;

        // Combine the individual rotation matrices directly
        R(0, 0) = cy * cp;
        R(0, 1) = cy * sp * sr - sy * cr;
        R(0, 2) = cy * sp * cr + sy * sr;
        R(1, 0) = sy * cp;
        R(1, 1) = sy * sp * sr + cy * cr;
        R(1, 2) = sy * sp * cr - cy * sr;
        R(2, 0) = -sp;
        R(2, 1) = cp * sr;
        R(2, 2) = cp * cr;

        return R;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
    {
        const double g_norm = g.norm();
        const Eigen::Vector3d ng1 = (std::abs(g_norm - 1.0) < 1e-6) ? g : (g / g_norm);

        static const Eigen::Vector3d ng2(0, 0, 1.0);
        const double dot = ng1.dot(ng2);

        if (dot > 0.99999)
        {
            // Vectors are nearly identical - return identity
            return Eigen::Matrix3d::Identity();
        }
        else if (dot < -0.99999)
        {
            // Vectors are nearly opposite -
            // Use an axis perpendicular to ng1 to rotate around
            Eigen::Vector3d axis = Eigen::Vector3d(1, 0, 0).cross(ng1);
            if (axis.norm() < 1e-6)
            {
                // If ng1 is aligned with x-axis, use y-axis instead
                axis = Eigen::Vector3d(0, 1, 0).cross(ng1);
            }
            axis.normalize();

            // 180 degree rotation around axis
            Eigen::AngleAxisd rotation(M_PI, axis);
            Eigen::Matrix3d R0 = rotation.toRotationMatrix();

            double yaw = Utility::R2ypr(R0).x();
            return Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        }
        else
        {
            // Normal case - compute rotation directly
            const Eigen::Vector3d axis = ng1.cross(ng2).normalized();
            const double angle = std::acos(dot);

            Eigen::AngleAxisd rotation(angle, axis);
            Eigen::Matrix3d R0 = rotation.toRotationMatrix();

            // Extract yaw and compensate
            double yaw = Utility::R2ypr(R0).x();
            return Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        }
    }

    template <size_t N>
    struct uint_
    {
    };

    template <size_t N, typename Lambda, typename IterT>
    static inline void unroller(const Lambda &f, const IterT &iter, uint_<N>)
    {
        unroller(f, iter, uint_<N - 1>());
        f(iter + N);
    }

    template <typename Lambda, typename IterT>
    static inline void unroller(const Lambda &f, const IterT &iter, uint_<0>)
    {
        f(iter);
    }

    template <typename T>
    static inline T normalizeAngle(const T &angle_degrees)
    {
        static constexpr T TWO_PI(360.0);
        static constexpr T PI(180.0);

        if (angle_degrees > T(0))
            return angle_degrees - TWO_PI * std::floor((angle_degrees + PI) / TWO_PI);
        else
            return angle_degrees + TWO_PI * std::floor((-angle_degrees + PI) / TWO_PI);
    }
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};