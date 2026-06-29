/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <eigen3/Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>

extern camodocal::CameraPtr m_camera;
extern Eigen::Vector3d tic;
extern Eigen::Matrix3d qic;
extern rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_match_img;
extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern std::string BRIEF_PATTERN_FILE;
extern std::string POSE_GRAPH_SAVE_PATH;
extern int ROW;
extern int COL;
extern int DEBUG_IMAGE;
extern double FOCAL_LENGTH_PX;

// Loop closure tunables.
extern double max_focallength;
extern double MIN_SCORE;
extern double PNP_INFLATION;
extern int RECALL_IGNORE_RECENT_COUNT;
extern double MAX_THETA_DIFF;
extern double MAX_POS_DIFF;
extern int MIN_LOOP_NUM;

// 4-DoF pose-graph optimization budget (bounds wall-time on long sequences).
extern int POSE_GRAPH_OPT_WINDOW;         // max keyframes optimized; 0 = unbounded (full graph)
extern double POSE_GRAPH_MAX_SOLVER_TIME; // ceres max_solver_time_in_seconds
extern int POSE_GRAPH_MAX_ITERATIONS;     // ceres max_num_iterations
extern int POSE_GRAPH_NUM_THREADS;        // ceres num_threads for the 4-DoF solve

// AnyLoc global-VPR loop-closure mode (DINOv2 ViT-S + VLAD).
extern bool USE_DL_LOOP;
extern std::string DESCRIPTOR_TOPIC; // topic carrying FrameDescriptors
extern double DL_LOOP_SIM_THRESH;    // cosine gate on the DINO-VLAD retrieval
extern double DL_RATIO_TEST;         // Lowe ratio for NN descriptor match
