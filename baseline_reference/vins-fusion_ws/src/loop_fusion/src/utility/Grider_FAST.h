/**
 * MIT License
 * Copyright (c) 2018 Patrick Geneva @ University of Delaware (Robot Perception & Navigation Group)
 * Copyright (c) 2018 Kevin Eckenhoff @ University of Delaware (Robot Perception & Navigation Group)
 * Copyright (c) 2018 Guoquan Huang @ University of Delaware (Robot Perception & Navigation Group)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 */

// Vendored verbatim from DL-VINS loop_fusion to give VINS-Fusion the same
// grid-capped FAST extraction (fair eval). With grid 1x1 + num_features=200
// this is: FAST over the whole image, keep the 200 strongest by response.

#ifndef LOOP_FUSION_GRIDER_FAST_H_
#define LOOP_FUSION_GRIDER_FAST_H_

#include <algorithm>
#include <cassert>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

class Grider_FAST
{
public:
    static bool compare_response(cv::KeyPoint first, cv::KeyPoint second)
    {
        return first.response > second.response;
    }

    static void perform_griding(const cv::Mat &img, std::vector<cv::KeyPoint> &pts, int num_features,
                                int grid_x, int grid_y, int threshold, bool nonmaxSuppression)
    {
        int size_x = img.cols / grid_x;
        int size_y = img.rows / grid_y;

        assert(size_x > 0);
        assert(size_y > 0);

        auto num_features_grid = (int)(num_features / (grid_x * grid_y)) + 1;

        for (int x = 0; x < img.cols; x += size_x)
        {
            for (int y = 0; y < img.rows; y += size_y)
            {
                if (x + size_x > img.cols || y + size_y > img.rows)
                    continue;

                cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);

                std::vector<cv::KeyPoint> pts_new;
                cv::FAST(img(img_roi), pts_new, threshold, nonmaxSuppression);

                std::sort(pts_new.begin(), pts_new.end(), Grider_FAST::compare_response);

                for (size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size(); i++)
                {
                    cv::KeyPoint pt_cor = pts_new.at(i);
                    pt_cor.pt.x += (float)x;
                    pt_cor.pt.y += (float)y;
                    pts.push_back(pt_cor);
                }
            }
        }
    }
};

#endif // LOOP_FUSION_GRIDER_FAST_H_
