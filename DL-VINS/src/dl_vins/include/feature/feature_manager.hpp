/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_HPP_
#define FEATURE_MANAGER_HPP_

#include "feature_struct.hpp"

#include <list>
#include <set>
#include <algorithm>

constexpr double INIT_DEPTH = 5.0;     // Legacy default depth, why 5?
constexpr double MIN_DEPTH = 0.1;      // Safety floor: triangulations / optimized depths below this are degenerate.
constexpr double MAX_DEPTH = 100.0;    // Safety ceiling: optimized depths above this are culled as degenerate.
constexpr int WINDOW_SIZE = 10;        // Compile-time sliding-window length.
constexpr double REF_FOCAL_PX = 460.0; // Reference focal (px) for normalized gates.

class FeatureManager
{
public:
    FeatureManager(double focal_length, double min_parallax_ratio);

    void setRic(const Eigen::Matrix3d ric[]);
    void clearState();

    // Add features from tracker output and decide keyframe vs non-keyframe
    bool addFeatureCheckParallax(int frame_count, const ObservationsMap &observations, double td);

    // Get corresponding point pairs between two frames
    FeatureCorrespondences getCorresponding(int frame_l, int frame_r) const;

    // Depth management
    void clearDepth();

    // Triangulation
    void triangulate(int frame_count,
                     Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                     Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

    // PnP
    void initFramePoseByPnP(int frame_count,
                            Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                            Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

    static bool solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P,
                               Point2fVec &pts2D, Point3fVec &pts3D);

    // Sliding window management
    void removeBack();
    void removeBackShiftDepth(const Eigen::Matrix3d &marg_R, const Eigen::Vector3d &marg_P,
                              const Eigen::Matrix3d &new_R, const Eigen::Vector3d &new_P);
    void removeFront(int frame_count);
    void removeOutlier(std::set<int> &outlier_index);

    // Access
    const std::map<int, FeatureTrack> &getFeatures() const { return features_; }
    // Used by the estimator for keyframe selection / outlier diagnostics.
    double lastAverageParallax() const { return last_average_parallax_; }
    int lastTrackNum() const { return last_track_num_; }

private:
    double compensatedParallax(const FeatureTrack &track, int frame_count) const;

    void triangulateStereo(FeatureTrack &track,
                           Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                           Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void triangulateTemporal(FeatureTrack &track,
                             Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[],
                             Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

    std::map<int, FeatureTrack> features_;
    Mat3dArr ric_;

    double focal_length_;
    double min_parallax_ratio_;

    // Statistics from last addFeatureCheckParallax call
    int last_track_num_ = 0;
    int new_feature_num_ = 0;
    int long_track_num_ = 0;
    double last_average_parallax_ = 0.0;
};

#endif // FEATURE_MANAGER_HPP_
