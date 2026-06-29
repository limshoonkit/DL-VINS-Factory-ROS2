#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace uosm::loop_fusion
{
    // Fixed-vocabulary VLAD aggregator.
    class VladCodebook
    {
    public:
        bool ready() const { return ready_; }
        int clusters() const { return clusters_; }
        int descriptorDim() const { return dim_; }
        int vladDim() const { return clusters_ * dim_; }

        // Load a fixed vocabulary from disk.
        // Binary layout: int32 clusters | int32 dim | clusters*dim float32 (row-major).
        bool load(const std::string &path);

        // Replace centres in-place (clusters x dim, CV_32F, row-major).
        bool setCenters(const cv::Mat &centers);

        // descriptors: N x dim, CV_32F (DINO patch tokens, or any local set).
        // Returns (clusters*dim) VLAD vector — per-cluster residual sums,
        // intra-normalised, signed-sqrt power-law, globally L2-normalised.
        std::vector<float> encode(const cv::Mat &descriptors) const;

    private:
        int clusters_ = 0;
        int dim_ = 0;
        std::vector<float> centers_; // clusters_ * dim_, row-major
        bool ready_ = false;
    };
} // namespace uosm::loop_fusion
