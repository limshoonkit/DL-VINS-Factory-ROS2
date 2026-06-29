#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <map>
#include <array>

struct Observation
{
    // This replaces Eigen::Matrix<double, 7, 1> point;
    Eigen::Vector3d point_c;  // Normalized coordinates in camera frame (x, y, 1)
    Eigen::Vector2d uv;       // Pixel coordinates
    Eigen::Vector2d velocity; // Pixel velocity
    double cur_td = 0.0;      // Time offset
    // RaCo learned covariance
    bool has_cov = false;
    Eigen::Matrix2d sqrt_info_px = Eigen::Matrix2d::Zero();
};

struct StereoObservation
{
    Observation left_obs;
    Observation right_obs;
    bool is_stereo = false;

    const Observation &left() const { return left_obs; }
    const Observation &right() const { return right_obs; }

    void setRight(const Observation &obs)
    {
        right_obs = obs;
        is_stereo = true;
    }
};

struct FeatureTrack
{
    int start_frame = 0;
    std::vector<StereoObservation> observations;
    double estimated_depth = -1.0;
    bool has_valid_depth = false; // set true once optimized to a positive (valid) depth

    int endFrame() const { return start_frame + static_cast<int>(observations.size()) - 1; }
};

using CameraObservation = std::pair<int, Observation>;
using ObservationsMap = std::map<int, std::vector<CameraObservation>>;
using FeatureCorrespondences = std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>;
using Pose3x4 = Eigen::Matrix<double, 3, 4>;
using Point2fVec = std::vector<cv::Point2f>;
using Point3fVec = std::vector<cv::Point3f>;
using Point2fMap = std::map<int, cv::Point2f>;
using Mat3dArr = std::array<Eigen::Matrix3d, 2>;