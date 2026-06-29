#ifndef UOSM_UTILITY_DEBUG_VISUALIZER_HPP_
#define UOSM_UTILITY_DEBUG_VISUALIZER_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <map>
#include <string>
#include <vector>

namespace uosm
{
    namespace utility
    {
        class DebugVisualizer
        {
        public:
            using Point2fVec = std::vector<cv::Point2f>;
            using Point2fMap = std::map<int, cv::Point2f>;

            struct LkFrameInputs
            {
                const cv::Mat &left_img;
                const cv::Mat &right_img; // empty for mono
                const Point2fVec &cur_pts_left;
                const Point2fVec &cur_pts_right; // empty for mono
                const std::vector<int> &ids_left;
                const std::vector<int> &track_cnt_left;
                const Point2fMap &prev_pts_by_id_left; // for arrow flow
            };

            struct LkStats
            {
                int tracked = 0;
                int new_features = 0;
                int stereo_matched = 0;
                int stereo_candidates = 0;
                int mask_culled = 0;
                bool fisheye = false;
            };

            // Render the LK / GFTT tracking visualization frame.
            static cv::Mat renderLkFrame(const LkFrameInputs &in, const LkStats &stats);

            struct StereoMatch
            {
                int idx_left = -1;
                int idx_right = -1;
                int outcome = 0;
            };

            struct DescriptorFrameInputs
            {
                const cv::Mat &left_img;
                const cv::Mat &right_img; // empty for mono
                const Point2fVec &tracked_pts_left;
                const std::vector<int> &tracked_ids_left; // for temporal arrows
                const std::vector<int> &tracked_track_cnt_left;
                const Point2fMap &prev_pts_by_id_left; // for temporal arrows
                const Point2fVec &dl_kps_left;
                const Point2fVec &dl_kps_right; // empty for mono
                const std::vector<StereoMatch> &stereo_matches;
            };

            struct DescriptorStats
            {
                int tracked = 0;
                int stereo_candidates = 0;
                int stereo_admitted = 0;
                int stereo_rej_depth = 0;
                int stereo_rej_reproj = 0;
                int temporal_matches = 0;
                float matcher_ms = 0.f;
                int frame_idx = 0;
                int mask_culled = 0;
                bool fisheye = false;
            };

            static cv::Mat renderDescriptorFrame(const DescriptorFrameInputs &in,
                                                 const DescriptorStats &stats);
        };
    } // namespace utility
} // namespace uosm

#endif // UOSM_UTILITY_DEBUG_VISUALIZER_HPP_
