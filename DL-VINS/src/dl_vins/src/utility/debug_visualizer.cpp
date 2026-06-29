#include "utility/debug_visualizer.hpp"

#include <cstdio>

namespace uosm
{
    namespace utility
    {
        namespace
        {
            void drawTextWithBackground(cv::Mat &canvas, const std::string &text, int x, int y,
                                        double font_scale = 0.55, int thickness = 1)
            {
                int baseline = 0;
                cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                              font_scale, thickness, &baseline);
                cv::rectangle(canvas, cv::Point(x - 2, y - ts.height - 2),
                              cv::Point(x + 2 + ts.width, y + baseline + 2),
                              cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(canvas, text, cv::Point(x, y),
                            cv::FONT_HERSHEY_SIMPLEX, font_scale,
                            cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
            }

            cv::Mat composeSideBySide(const cv::Mat &left, const cv::Mat &right)
            {
                cv::Mat canvas;
                if (!right.empty())
                    cv::hconcat(left, right, canvas);
                else
                    canvas = left.clone();
                if (canvas.channels() == 1)
                    cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
                return canvas;
            }
        } // namespace

        cv::Mat DebugVisualizer::renderLkFrame(const LkFrameInputs &in, const LkStats &stats)
        {
            if (in.left_img.empty())
                return cv::Mat();

            cv::Mat canvas = composeSideBySide(in.left_img, in.right_img);
            const int W = in.left_img.cols;

            // Tracked points on left, color blended by track age.
            for (size_t i = 0; i < in.cur_pts_left.size(); ++i)
            {
                double len = std::min(1.0, 1.0 * in.track_cnt_left[i] / 20.0);
                cv::circle(canvas, in.cur_pts_left[i], 2,
                           cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }

            // Right-camera tracked points (LK stereo).
            if (!in.right_img.empty())
            {
                for (const auto &p : in.cur_pts_right)
                {
                    cv::Point2f rp(p.x + W, p.y);
                    cv::circle(canvas, rp, 2, cv::Scalar(0, 255, 0), 2);
                }
            }

            // Forward-flow arrows on left.
            for (size_t i = 0; i < in.ids_left.size(); ++i)
            {
                auto it = in.prev_pts_by_id_left.find(in.ids_left[i]);
                if (it != in.prev_pts_by_id_left.end())
                {
                    cv::arrowedLine(canvas, in.cur_pts_left[i], it->second,
                                    cv::Scalar(0, 255, 0), 1, cv::LINE_8, 0, 0.2f);
                }
            }

            const int line_h = 22;
            const int y_top = 25;
            int y = y_top;
            char buf[128];
            std::snprintf(buf, sizeof(buf), "Tracked: %d | New: %d", stats.tracked, stats.new_features);
            drawTextWithBackground(canvas, buf, 10, y);
            // Stereo line is meaningless in mono (no right image); skip it.
            if (!in.right_img.empty())
            {
                y += line_h;
                std::snprintf(buf, sizeof(buf), "Stereo: %d/%d", stats.stereo_matched, stats.stereo_candidates);
                drawTextWithBackground(canvas, buf, 10, y);
            }
            // FOV-mask cull count, only when a fisheye/FOV mask is active.
            if (stats.fisheye)
            {
                y += line_h;
                std::snprintf(buf, sizeof(buf), "Mask culled: %d", stats.mask_culled);
                drawTextWithBackground(canvas, buf, 10, y);
            }

            return canvas;
        }

        cv::Mat DebugVisualizer::renderDescriptorFrame(const DescriptorFrameInputs &in,
                                                       const DescriptorStats &stats)
        {
            if (in.left_img.empty())
                return cv::Mat();

            cv::Mat canvas = composeSideBySide(in.left_img, in.right_img);
            const int W = in.left_img.cols;

            const cv::Scalar COL_DL_DOT(40, 160, 40);
            const cv::Scalar COL_ADMIT(0, 255, 0);

            // Raw DL keypoints — detection density.
            for (const auto &p : in.dl_kps_left)
                cv::circle(canvas, p, 1, COL_DL_DOT, cv::FILLED);
            if (!in.right_img.empty())
            {
                for (const auto &p : in.dl_kps_right)
                {
                    cv::Point2f rp(p.x + W, p.y);
                    cv::circle(canvas, rp, 1, COL_DL_DOT, cv::FILLED);
                }
            }

            // Temporal flow arrows (cur -> prev) on the left.
            for (size_t i = 0; i < in.tracked_pts_left.size() && i < in.tracked_ids_left.size(); ++i)
            {
                auto it = in.prev_pts_by_id_left.find(in.tracked_ids_left[i]);
                if (it != in.prev_pts_by_id_left.end())
                    cv::arrowedLine(canvas, in.tracked_pts_left[i], it->second,
                                    cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0, 0.25f);
            }

            // Tracked-feature markers (left), color blended with track age.
            for (size_t i = 0; i < in.tracked_pts_left.size(); ++i)
            {
                double len = std::min(1.0, 1.0 * in.tracked_track_cnt_left[i] / 20.0);
                cv::circle(canvas, in.tracked_pts_left[i], 2,
                           cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
            }

            // Stereo correspondences as overlay only (admitted matches only).
            for (const auto &t : in.stereo_matches)
            {
                if (t.outcome != 0)
                    continue;
                if (t.idx_left >= 0 && t.idx_left < static_cast<int>(in.dl_kps_left.size()))
                    cv::circle(canvas, in.dl_kps_left[t.idx_left], 4, COL_ADMIT, 1, cv::LINE_AA);
                if (!in.right_img.empty() && t.idx_right >= 0 &&
                    t.idx_right < static_cast<int>(in.dl_kps_right.size()))
                {
                    cv::Point2f pR(in.dl_kps_right[t.idx_right].x + W,
                                   in.dl_kps_right[t.idx_right].y);
                    cv::circle(canvas, pR, 4, COL_ADMIT, 1, cv::LINE_AA);
                }
            }

            const int line_h = 20;
            int y = line_h;
            char buf[256];

            std::snprintf(buf, sizeof(buf), "LG temporal: tracked=%d matches=%d",
                          stats.tracked, stats.temporal_matches);
            drawTextWithBackground(canvas, buf, 8, y, 0.5);

            // Stereo line is meaningless in mono (no right image); skip it.
            if (!in.right_img.empty())
            {
                y += line_h;
                std::snprintf(buf, sizeof(buf),
                              "Stereo: cand=%d admit=%d rej D/R=%d/%d",
                              stats.stereo_candidates, stats.stereo_admitted,
                              stats.stereo_rej_depth,
                              stats.stereo_rej_reproj);
                drawTextWithBackground(canvas, buf, 8, y, 0.5);
            }
            // FOV-mask cull count, only when a fisheye/FOV mask is active.
            if (stats.fisheye)
            {
                y += line_h;
                std::snprintf(buf, sizeof(buf), "Mask culled: %d", stats.mask_culled);
                drawTextWithBackground(canvas, buf, 8, y, 0.5);
            }

            return canvas;
        }
    } // namespace utility
} // namespace uosm
