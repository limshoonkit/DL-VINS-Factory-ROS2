/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "../../include/feature/feature_manager.hpp"
#include "../../include/feature/triangulation.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <rclcpp/logging.hpp>

FeatureManager::FeatureManager(double focal_length,
                               double min_parallax_ratio)
    : focal_length_(focal_length),
      min_parallax_ratio_(min_parallax_ratio)
{
    ric_[0].setIdentity();
    ric_[1].setIdentity();
}

void FeatureManager::setRic(const Eigen::Matrix3d ric[])
{
    ric_[0] = ric[0];
    ric_[1] = ric[1];
}

void FeatureManager::clearState()
{
    features_.clear();
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const ObservationsMap &observations, double td)
{
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num_ = 0;
    last_average_parallax_ = 0;
    new_feature_num_ = 0;
    long_track_num_ = 0;

    for (const auto &[feature_id, cam_obs_vec] : observations)
    {
        StereoObservation stereo_obs;

        for (const auto &[cam_id, obs] : cam_obs_vec)
        {
            if (cam_id == 0)
            {
                stereo_obs.left_obs = obs;
                stereo_obs.left_obs.cur_td = td;
            }
            else if (cam_id == 1)
            {
                stereo_obs.setRight(obs);
                stereo_obs.right_obs.cur_td = td;
            }
        }

        auto it = features_.find(feature_id);
        if (it == features_.end())
        {
            // New feature
            FeatureTrack track;
            track.start_frame = frame_count;
            track.observations.push_back(stereo_obs);
            features_[feature_id] = std::move(track);
            new_feature_num_++;
        }
        else
        {
            // Existing feature - add new observation
            it->second.observations.push_back(stereo_obs);
            last_track_num_++;
            if (static_cast<int>(it->second.observations.size()) >= 4)
                long_track_num_++;
        }
    }

    if (frame_count < 2 || last_track_num_ < 20 || long_track_num_ < 40 ||
        new_feature_num_ > 0.5 * last_track_num_)
        return true;

    for (const auto &[id, track] : features_)
    {
        if (track.start_frame <= frame_count - 2 &&
            track.endFrame() >= frame_count - 1)
        {
            parallax_sum += compensatedParallax(track, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }

    last_average_parallax_ = parallax_sum / parallax_num * focal_length_;
    return parallax_sum / parallax_num >= min_parallax_ratio_;
}

FeatureCorrespondences FeatureManager::getCorresponding(int frame_t0, int frame_t1) const
{
    FeatureCorrespondences corres;
    for (const auto &[id, track] : features_)
    {
        if (track.start_frame <= frame_t0 && track.endFrame() >= frame_t1)
        {
            int idx_t0 = frame_t0 - track.start_frame;
            int idx_t1 = frame_t1 - track.start_frame;

            Eigen::Vector3d a = track.observations[idx_t0].left().point_c;
            Eigen::Vector3d b = track.observations[idx_t1].left().point_c;

            corres.emplace_back(a, b);
        }
    }
    return corres;
}

void FeatureManager::clearDepth()
{
    for (auto &[id, track] : features_)
        track.estimated_depth = -1.0;
}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P,
                                    Point2fVec &pts2D, Point3fVec &pts3D)
{
    // w_T_cam ---> cam_T_w
    Eigen::Matrix3d R_initial = R.inverse();
    Eigen::Vector3d P_initial = -(R_initial * P);

    if (static_cast<int>(pts2D.size()) < 4)
        return false;

    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

    bool pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);

    if (!pnp_succ)
        return false;

    cv::Rodrigues(rvec, r);
    Eigen::MatrixXd R_pnp, T_pnp;
    cv::cv2eigen(r, R_pnp);
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

void FeatureManager::initFramePoseByPnP(int frame_count,
                                        Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                                        Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
{
    if (frame_count <= 0)
        return;

    Point2fVec pts2D;
    Point3fVec pts3D;

    for (const auto &[id, track] : features_)
    {
        if (track.estimated_depth <= 0)
            continue;

        int index = frame_count - track.start_frame;
        if (static_cast<int>(track.observations.size()) < index + 1)
            continue;

        Eigen::Vector3d ptsInCam = ric[0] * (track.observations[0].left().point_c * track.estimated_depth) + tic[0];
        Eigen::Vector3d ptsInWorld = Rs[track.start_frame] * ptsInCam + Ps[track.start_frame];

        pts3D.emplace_back(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());

        const auto &obs = track.observations[index].left();
        pts2D.emplace_back(obs.point_c.x(), obs.point_c.y());
    }

    Eigen::Matrix3d RCam = Rs[frame_count - 1] * ric[0];
    Eigen::Vector3d PCam = Rs[frame_count - 1] * tic[0] + Ps[frame_count - 1];

    if (solvePoseByPnP(RCam, PCam, pts2D, pts3D))
    {
        Rs[frame_count] = RCam * ric[0].transpose();
        Ps[frame_count] = -RCam * ric[0].transpose() * tic[0] + PCam;
        Eigen::Quaterniond Q(Rs[frame_count]);
        RCLCPP_DEBUG(rclcpp::get_logger("feature_manager"),
                     "frame_count: %d pnp Q [%.4f, %.4f, %.4f, %.4f] P [%.4f, %.4f, %.4f]",
                     frame_count, Q.w(), Q.x(), Q.y(), Q.z(),
                     Ps[frame_count].x(), Ps[frame_count].y(), Ps[frame_count].z());
    }
}

void FeatureManager::triangulate(int frame_count,
                                 Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                                 Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
{
    for (auto &[id, track] : features_)
    {
        if (track.estimated_depth > 0)
            continue;

        if (track.observations.front().is_stereo)
        {
            triangulateStereo(track, Ps, Rs, tic, ric);
        }
        else
        {
            triangulateTemporal(track, Ps, Rs, tic, ric);
        }
    }
}

void FeatureManager::triangulateStereo(FeatureTrack &track,
                                       Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                                       Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
{
    const int imu_i = track.start_frame;

    const Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
    const Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
    Pose3x4 leftPose;
    leftPose.leftCols<3>() = R0.transpose();
    leftPose.rightCols<1>() = -R0.transpose() * t0;

    const Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
    const Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
    Pose3x4 rightPose;
    rightPose.leftCols<3>() = R1.transpose();
    rightPose.rightCols<1>() = -R1.transpose() * t1;

    Eigen::Vector3d point3d;
    triangulation::triangulatePoint(leftPose, rightPose,
                                    track.observations[0].left().point_c.head(2),
                                    track.observations[0].right().point_c.head(2),
                                    point3d);

    const double depth = (leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>()).z();
    track.estimated_depth = (depth > MIN_DEPTH && depth < MAX_DEPTH) ? depth : INIT_DEPTH;
}

void FeatureManager::triangulateTemporal(FeatureTrack &track,
                                         Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                                         Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
{
    const int imu_i = track.start_frame;
    const Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
    const Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];

    if (track.observations.size() >= 4)
    {
        // Multi-view SVD across all observations
        Eigen::MatrixXd svd_A(2 * static_cast<int>(track.observations.size()), 4);
        int svd_idx = 0;

        for (size_t k = 0; k < track.observations.size(); ++k)
        {
            const int imu_j = imu_i + static_cast<int>(k);
            const Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            const Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            const Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            const Eigen::Matrix3d R = R0.transpose() * R1;

            Pose3x4 P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;

            const Eigen::Vector3d f = track.observations[k].left().point_c.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
        }

        const Eigen::Vector4d svd_V =
            Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        const double depth = svd_V[2] / svd_V[3];

        track.estimated_depth = (depth > MIN_DEPTH) ? depth : INIT_DEPTH;
    }
    else if (track.observations.size() > 1)
    {
        // 2-frame fallback
        const Eigen::Vector3d t1 = Ps[imu_i + 1] + Rs[imu_i + 1] * tic[0];
        const Eigen::Matrix3d R1 = Rs[imu_i + 1] * ric[0];

        Pose3x4 leftPose, rightPose;
        leftPose.leftCols<3>() = R0.transpose();
        leftPose.rightCols<1>() = -R0.transpose() * t0;
        rightPose.leftCols<3>() = R1.transpose();
        rightPose.rightCols<1>() = -R1.transpose() * t1;

        Eigen::Vector3d point3d;
        triangulation::triangulatePoint(leftPose, rightPose,
                                        track.observations[0].left().point_c.head(2),
                                        track.observations[1].left().point_c.head(2),
                                        point3d);

        const double depth = (leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>()).z();

        track.estimated_depth = (depth > MIN_DEPTH) ? depth : INIT_DEPTH;
    }
}

void FeatureManager::removeOutlier(std::set<int> &outlier_index)
{
    for (int idx : outlier_index)
        features_.erase(idx);
}

void FeatureManager::removeBackShiftDepth(const Eigen::Matrix3d &marg_R, const Eigen::Vector3d &marg_P,
                                          const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_P)
{
    for (auto it = features_.begin(); it != features_.end();)
    {
        auto &track = it->second;

        if (track.start_frame != 0)
        {
            track.start_frame--;
            ++it;
        }
        else
        {
            Eigen::Vector3d uv_i = track.observations[0].left().point_c;
            track.observations.erase(track.observations.begin());

            if (track.observations.size() < 2)
            {
                it = features_.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * track.estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                track.estimated_depth = (dep_j > 0) ? dep_j : INIT_DEPTH;
                ++it;
            }
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = features_.begin(); it != features_.end();)
    {
        auto &track = it->second;

        if (track.start_frame != 0)
        {
            track.start_frame--;
            ++it;
        }
        else
        {
            track.observations.erase(track.observations.begin());
            if (track.observations.empty())
                it = features_.erase(it);
            else
                ++it;
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = features_.begin(); it != features_.end();)
    {
        auto &track = it->second;

        if (track.start_frame == frame_count)
        {
            track.start_frame--;
            ++it;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - track.start_frame;
            if (track.endFrame() < frame_count - 1)
            {
                ++it;
                continue;
            }
            track.observations.erase(track.observations.begin() + j);
            if (track.observations.empty())
                it = features_.erase(it);
            else
                ++it;
        }
    }
}

double FeatureManager::compensatedParallax(const FeatureTrack &track, int frame_count) const
{
    // Parallax between second-last and third-last frame
    const auto &frame_i = track.observations[frame_count - 2 - track.start_frame];
    const auto &frame_j = track.observations[frame_count - 1 - track.start_frame];

    Eigen::Vector3d p_j = frame_j.left().point_c;
    double u_j = p_j(0);
    double v_j = p_j(1);

    Eigen::Vector3d p_i = frame_i.left().point_c;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j;
    double dv = v_i - v_j;

    // Without rotation compensation (simplified)
    Eigen::Vector3d p_i_comp = p_i;
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j;
    double dv_comp = v_i_comp - v_j;

    return std::sqrt(std::min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp));
}
