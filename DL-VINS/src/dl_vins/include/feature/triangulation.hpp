#pragma once

#include "feature_struct.hpp"

#include <algorithm>
#include <cmath>

namespace triangulation
{
    inline Eigen::Vector4d triangulateFast(const Eigen::Vector3d &p1,
                                           const Eigen::Vector3d &e1,
                                           const Eigen::Vector3d &p2,
                                           const Eigen::Vector3d &e2,
                                           double sigma,
                                           bool &isValid,
                                           bool &isParallel)
    {
        isParallel = false;
        isValid = true;

        const Eigen::Vector3d t12 = p2 - p1;
        Eigen::Vector2d b;
        b[0] = t12.dot(e1);
        b[1] = t12.dot(e2);
        Eigen::Matrix2d A;
        A(0, 0) = e1.dot(e1);
        A(1, 0) = e1.dot(e2);
        A(0, 1) = -A(1, 0);
        A(1, 1) = -e2.dot(e2);

        bool invertible = false;
        Eigen::Matrix2d A_inverse;
        A.computeInverseWithCheck(A_inverse, invertible, 1.0e-12);
        Eigen::Vector2d lambda = Eigen::Vector2d::Zero();
        if (invertible)
            lambda = A_inverse * b;

        const double cos_2p6_sigma = std::cos(2.6 * sigma);
        const double cos_6_sigma = std::cos(6.0 * sigma);

        auto parallelCase = [&]()
        {
            isParallel = true;
            Eigen::Vector4d hp(0.0, 0.0, 0.0, 1.0);
            isValid = true;
            const Eigen::Vector3d m = p1 + 0.5 * t12;
            const Eigen::Vector3d midpoint =
                m + 40.0 * std::max(0.01, t12.norm()) * (e1 + e2);
            hp.head<3>() = midpoint;

            if (e1.dot((midpoint - p1).normalized()) < cos_2p6_sigma)
                isValid = false;
            if (e2.dot((midpoint - p2).normalized()) < cos_2p6_sigma)
                isValid = false;
            return hp;
        };

        if (!invertible)
        {
            return parallelCase();
        }
        if (lambda[0] < 0.01 || lambda[1] < 0.01)
        {
            return parallelCase();
        }

        const Eigen::Vector3d xm = lambda[0] * e1 + p1;
        const Eigen::Vector3d xn = lambda[1] * e2 + p2;
        const Eigen::Vector3d midpoint = 0.5 * (xm + xn);

        if (e1.dot((midpoint - p1).normalized()) < cos_2p6_sigma)
            isValid = false;
        if (e2.dot((midpoint - p2).normalized()) < cos_2p6_sigma)
            isValid = false;
        if ((midpoint - p2).normalized().dot((midpoint - p1).normalized()) > cos_6_sigma)
            isParallel = true;

        return Eigen::Vector4d(midpoint.x(), midpoint.y(), midpoint.z(), 1.0);
    }

    inline void triangulatePoint(const Pose3x4 &Pose0, const Pose3x4 &Pose1,
                                 const Eigen::Vector2d &point0, const Eigen::Vector2d &point1,
                                 Eigen::Vector3d &point_3d)
    {
        Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
        design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
        design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
        design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
        design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);

        Eigen::Vector4d triangulated_point =
            design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

        point_3d(0) = triangulated_point(0) / triangulated_point(3);
        point_3d(1) = triangulated_point(1) / triangulated_point(3);
        point_3d(2) = triangulated_point(2) / triangulated_point(3);
    }
} // namespace triangulation
