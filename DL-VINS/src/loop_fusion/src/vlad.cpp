#include "loop_fusion/vlad.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>

#include <opencv2/core.hpp>

namespace uosm::loop_fusion
{
    bool VladCodebook::load(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
        {
            std::cerr << "[vlad] cannot open codebook: " << path << std::endl;
            return false;
        }
        int32_t k = 0, d = 0;
        f.read(reinterpret_cast<char *>(&k), sizeof(int32_t));
        f.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
        if (!f || k <= 0 || d <= 0)
        {
            std::cerr << "[vlad] bad codebook header in " << path << std::endl;
            return false;
        }
        std::vector<float> buf(static_cast<size_t>(k) * static_cast<size_t>(d));
        f.read(reinterpret_cast<char *>(buf.data()),
               static_cast<std::streamsize>(buf.size() * sizeof(float)));
        if (!f)
        {
            std::cerr << "[vlad] truncated codebook: " << path << std::endl;
            return false;
        }
        centers_ = std::move(buf);
        clusters_ = k;
        dim_ = d;
        ready_ = true;
        std::cout << "[vlad] codebook loaded: " << k << " clusters x " << d
                  << " dim  (VLAD dim " << vladDim() << ")" << std::endl;
        return true;
    }

    bool VladCodebook::setCenters(const cv::Mat &centers)
    {
        if (centers.empty() || centers.type() != CV_32F ||
            centers.rows <= 0 || centers.cols <= 0)
        {
            std::cerr << "[vlad] setCenters: invalid matrix\n";
            return false;
        }
        clusters_ = centers.rows;
        dim_ = centers.cols;
        centers_.assign(static_cast<size_t>(clusters_) * dim_, 0.0f);
        for (int r = 0; r < clusters_; ++r)
        {
            const float *src = centers.ptr<float>(r);
            std::copy(src, src + dim_, centers_.begin() + static_cast<size_t>(r) * dim_);
        }
        ready_ = true;
        return true;
    }

    std::vector<float> VladCodebook::encode(const cv::Mat &descriptors) const
    {
        std::vector<float> vlad(static_cast<size_t>(clusters_) * static_cast<size_t>(dim_), 0.0f);
        if (!ready_ || descriptors.empty() || descriptors.cols != dim_ ||
            descriptors.type() != CV_32F)
            return vlad;

        // Accumulate per-cluster residual sums against the nearest centre.
        for (int i = 0; i < descriptors.rows; ++i)
        {
            const float *desc = descriptors.ptr<float>(i);
            int best = 0;
            float best_d2 = std::numeric_limits<float>::max();
            for (int c = 0; c < clusters_; ++c)
            {
                const float *ctr = &centers_[static_cast<size_t>(c) * dim_];
                float d2 = 0.0f;
                for (int j = 0; j < dim_; ++j)
                {
                    const float e = desc[j] - ctr[j];
                    d2 += e * e;
                }
                if (d2 < best_d2)
                {
                    best_d2 = d2;
                    best = c;
                }
            }
            float *acc = &vlad[static_cast<size_t>(best) * dim_];
            const float *ctr = &centers_[static_cast<size_t>(best) * dim_];
            for (int j = 0; j < dim_; ++j)
                acc[j] += desc[j] - ctr[j];
        }

        // Intra-normalisation (per cluster).
        for (int c = 0; c < clusters_; ++c)
        {
            float *acc = &vlad[static_cast<size_t>(c) * dim_];
            float nrm = 0.0f;
            for (int j = 0; j < dim_; ++j)
                nrm += acc[j] * acc[j];
            nrm = std::sqrt(nrm);
            if (nrm < 1e-12f)
                nrm = 1e-12f;
            for (int j = 0; j < dim_; ++j)
                acc[j] /= nrm;
        }

        // Signed-sqrt power-law + global L2 normalisation.
        float gnrm = 0.0f;
        for (float &v : vlad)
        {
            v = (v >= 0.0f ? 1.0f : -1.0f) * std::sqrt(std::fabs(v));
            gnrm += v * v;
        }
        gnrm = std::sqrt(gnrm);
        if (gnrm < 1e-12f)
            gnrm = 1e-12f;
        for (float &v : vlad)
            v /= gnrm;
        return vlad;
    }
} // namespace uosm::loop_fusion
