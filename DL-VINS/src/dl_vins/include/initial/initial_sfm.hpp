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
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <map>
#include <vector>

struct SFMFeature
{
    bool state = false;
    int id = -1;
    std::vector<std::pair<int, Eigen::Vector2d>> observation;
    double position[3]{};
    double depth = 0.0;
};

struct ReprojectionError3D
{
    ReprojectionError3D(double observed_u, double observed_v)
        : observed_u(observed_u), observed_v(observed_v) {}

    template <typename T>
    bool operator()(const T *const camera_R, const T *const camera_T,
                    const T *point, T *residuals) const
    {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction *Create(double observed_x, double observed_y)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(
            new ReprojectionError3D(observed_x, observed_y));
    }

    double observed_u;
    double observed_v;
};

class GlobalSFM
{
public:
    GlobalSFM() = default;

    /**
     * @brief Incremental SFM reconstruction
     * @param frame_num  Number of frames (WINDOW_SIZE + 1)
     * @param q          [out] Per-frame rotations (world from camera)
     * @param T          [out] Per-frame translations (world)
     * @param l          Anchor frame index (paired with frame_num-1)
     * @param relative_R Relative rotation from frame l to frame_num-1
     * @param relative_T Relative translation from frame l to frame_num-1
     * @param sfm_f      Feature observations across frames
     * @param sfm_tracked_points  [out] Reconstructed 3D points
     * @return true if reconstruction succeeded
     */
    bool construct(int frame_num, Eigen::Quaterniond *q, Eigen::Vector3d *T, int l,
                   const Eigen::Matrix3d relative_R, const Eigen::Vector3d relative_T,
                   std::vector<SFMFeature> &sfm_f,
                   std::map<int, Eigen::Vector3d> &sfm_tracked_points);

private:
    bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                         int i, std::vector<SFMFeature> &sfm_f);

    void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                              int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                              std::vector<SFMFeature> &sfm_f);

    int feature_num_ = 0;
};
