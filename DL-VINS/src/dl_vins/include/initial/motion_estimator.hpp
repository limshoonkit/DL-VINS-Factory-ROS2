/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

class MotionEstimator
{
public:
    void setFocalLength(double focal) { focal_length_ = focal; }

    /**
     * @brief Solve relative rotation and translation from feature correspondences
     * Uses fundamental matrix estimation (RANSAC) + essential matrix decomposition.
     * Points are in normalized camera coordinates.
     * @param corres  Matched point pairs (normalized coords)
     * @param R       [out] Relative rotation (R_{l->newest})
     * @param T       [out] Relative translation (unit vector)
     * @return true if >= 12 inliers found
     */
    bool solveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres,
                         Eigen::Matrix3d &R, Eigen::Vector3d &T);

private:
    double focal_length_ = 460.0;
};
