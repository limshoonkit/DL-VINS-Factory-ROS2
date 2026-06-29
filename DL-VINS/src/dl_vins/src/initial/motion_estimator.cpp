/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "../../include/initial/motion_estimator.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

bool MotionEstimator::solveRelativeRT(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres,
    Eigen::Matrix3d &Rotation, Eigen::Vector3d &Translation)
{
    if (corres.size() < 15)
        return false;

    std::vector<cv::Point2f> ll, rr;
    ll.reserve(corres.size());
    rr.reserve(corres.size());
    for (const auto &c : corres)
    {
        ll.emplace_back(static_cast<float>(c.first(0)), static_cast<float>(c.first(1)));
        rr.emplace_back(static_cast<float>(c.second(0)), static_cast<float>(c.second(1)));
    }

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat rot, trans;
    int inlier_cnt = 0;
    try
    {
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / focal_length_, 0.99, mask);
        if (E.empty() || E.rows != 3 || E.cols != 3)
            return false;
        inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
    }
    catch (const cv::Exception &e)
    {
        // try next frame
        return false;
    }
    if (rot.rows != 3 || rot.cols != 3 || trans.rows != 3)
        return false;

    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    for (int i = 0; i < 3; i++)
    {
        T(i) = trans.at<double>(i, 0);
        for (int j = 0; j < 3; j++)
            R(i, j) = rot.at<double>(i, j);
    }

    Rotation = R.transpose();
    Translation = -R.transpose() * T;

    return inlier_cnt > 12;
}
