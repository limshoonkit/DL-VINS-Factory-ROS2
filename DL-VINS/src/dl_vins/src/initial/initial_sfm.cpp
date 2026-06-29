/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "../../include/initial/initial_sfm.hpp"
#include "../../include/feature/triangulation.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <rclcpp/logging.hpp>

bool GlobalSFM::solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                                int i, std::vector<SFMFeature> &sfm_f)
{
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for (int j = 0; j < feature_num_; j++)
    {
        if (!sfm_f[j].state)
            continue;
        for (int k = 0; k < static_cast<int>(sfm_f[j].observation.size()); k++)
        {
            if (sfm_f[j].observation[k].first == i)
            {
                Eigen::Vector2d img_pts = sfm_f[j].observation[k].second;
                pts_2_vector.emplace_back(static_cast<float>(img_pts(0)),
                                          static_cast<float>(img_pts(1)));
                pts_3_vector.emplace_back(static_cast<float>(sfm_f[j].position[0]),
                                          static_cast<float>(sfm_f[j].position[1]),
                                          static_cast<float>(sfm_f[j].position[2]));
                break;
            }
        }
    }
    if (static_cast<int>(pts_2_vector.size()) < 15)
    {
        RCLCPP_WARN(rclcpp::get_logger("sfm"),
                    "Unstable feature tracking, only %zu points for PnP",
                    pts_2_vector.size());
        if (static_cast<int>(pts_2_vector.size()) < 10)
            return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

    // RANSAC PnP gives a pose estimate robust to outlier 2D-3D correspondences.
    if (!cv::solvePnPRansac(pts_3_vector, pts_2_vector, K, D, rvec, t,
                            /*useExtrinsicGuess=*/true,
                            /*iterationsCount=*/100,
                            /*reprojectionError=*/0.017f,
                            /*confidence=*/0.99,
                            cv::noArray(),
                            cv::SOLVEPNP_ITERATIVE))
        return false;

    cv::Rodrigues(rvec, r);
    Eigen::MatrixXd R_pnp, T_pnp;
    cv::cv2eigen(r, R_pnp);
    cv::cv2eigen(t, T_pnp);
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                                     int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                                     std::vector<SFMFeature> &sfm_f)
{
    assert(frame0 != frame1);
    for (int j = 0; j < feature_num_; j++)
    {
        if (sfm_f[j].state)
            continue;
        bool has_0 = false, has_1 = false;
        Eigen::Vector2d point0, point1;
        for (int k = 0; k < static_cast<int>(sfm_f[j].observation.size()); k++)
        {
            if (sfm_f[j].observation[k].first == frame0)
            {
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }
            if (sfm_f[j].observation[k].first == frame1)
            {
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }
        if (has_0 && has_1)
        {
            Eigen::Vector3d point_3d;
            triangulation::triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }
}

bool GlobalSFM::construct(int frame_num, Eigen::Quaterniond *q, Eigen::Vector3d *T, int l,
                          const Eigen::Matrix3d relative_R, const Eigen::Vector3d relative_T,
                          std::vector<SFMFeature> &sfm_f,
                          std::map<int, Eigen::Vector3d> &sfm_tracked_points)
{
    feature_num_ = static_cast<int>(sfm_f.size());

    // Initial two-view setup: frame l at origin, frame (frame_num-1) at relative pose
    q[l] = Eigen::Quaterniond::Identity();
    T[l].setZero();
    q[frame_num - 1] = q[l] * Eigen::Quaterniond(relative_R);
    T[frame_num - 1] = relative_T;

    // Allocate per-frame arrays (camera-frame convention: c_R * world_pt + c_T)
    std::vector<Eigen::Matrix3d> c_Rotation(frame_num);
    std::vector<Eigen::Vector3d> c_Translation(frame_num);
    std::vector<Eigen::Quaterniond> c_Quat(frame_num);
    std::vector<std::array<double, 4>> c_rotation(frame_num);
    std::vector<std::array<double, 3>> c_translation(frame_num);
    std::vector<Eigen::Matrix<double, 3, 4>> Pose(frame_num);

    // Set anchor frame l
    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1.0 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    // Set newest frame
    c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
    c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
    c_Translation[frame_num - 1] = -1.0 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
    Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // Forward: solve PnP for frames l+1..frame_num-2, triangulate with frame_num-1
    for (int i = l; i < frame_num - 1; i++)
    {
        if (i > l)
        {
            Eigen::Matrix3d R_initial = c_Rotation[i - 1];
            Eigen::Vector3d P_initial = c_Translation[i - 1];
            if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
                return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }
        triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
    }

    // Triangulate l with l+1..frame_num-2
    for (int i = l + 1; i < frame_num - 1; i++)
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

    // Backward: solve PnP for frames l-1..0, triangulate with l
    for (int i = l - 1; i >= 0; i--)
    {
        Eigen::Matrix3d R_initial = c_Rotation[i + 1];
        Eigen::Vector3d P_initial = c_Translation[i + 1];
        if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
    }

    // Triangulate remaining features
    for (int j = 0; j < feature_num_; j++)
    {
        if (sfm_f[j].state)
            continue;
        if (static_cast<int>(sfm_f[j].observation.size()) >= 2)
        {
            int frame_0 = sfm_f[j].observation.front().first;
            Eigen::Vector2d point0 = sfm_f[j].observation.front().second;
            int frame_1 = sfm_f[j].observation.back().first;
            Eigen::Vector2d point1 = sfm_f[j].observation.back().second;
            Eigen::Vector3d point_3d;
            triangulation::triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }

    // Full bundle adjustment
    ceres::Problem problem;
    ceres::Manifold *quaternion_manifold = new ceres::QuaternionManifold();

    for (int i = 0; i < frame_num; i++)
    {
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();
        problem.AddParameterBlock(c_rotation[i].data(), 4, quaternion_manifold);
        problem.AddParameterBlock(c_translation[i].data(), 3);
        if (i == l)
            problem.SetParameterBlockConstant(c_rotation[i].data());
        if (i == l || i == frame_num - 1)
            problem.SetParameterBlockConstant(c_translation[i].data());
    }

    // Robust loss on SFM reprojection residuals.
    auto *sfm_loss = new ceres::HuberLoss(0.005f);

    for (int i = 0; i < feature_num_; i++)
    {
        if (!sfm_f[i].state)
            continue;
        for (int j = 0; j < static_cast<int>(sfm_f[i].observation.size()); j++)
        {
            int frame_idx = sfm_f[i].observation[j].first;
            ceres::CostFunction *cost_function = ReprojectionError3D::Create(
                sfm_f[i].observation[j].second.x(),
                sfm_f[i].observation[j].second.y());
            problem.AddResidualBlock(cost_function, sfm_loss,
                                     c_rotation[frame_idx].data(),
                                     c_translation[frame_idx].data(),
                                     sfm_f[i].position);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.termination_type != ceres::CONVERGENCE && summary.final_cost >= 5e-03)
    {
        RCLCPP_WARN(rclcpp::get_logger("sfm"), "Vision-only BA did not converge");
        return false;
    }

    // Extract results: convert from camera frame back to world frame
    for (int i = 0; i < frame_num; i++)
    {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
    }
    for (int i = 0; i < frame_num; i++)
    {
        T[i] = -1.0 * (q[i] * Eigen::Vector3d(c_translation[i][0],
                                              c_translation[i][1],
                                              c_translation[i][2]));
    }
    for (int i = 0; i < static_cast<int>(sfm_f.size()); i++)
    {
        if (sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] =
                Eigen::Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
    }
    return true;
}