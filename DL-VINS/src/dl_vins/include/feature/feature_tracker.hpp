#ifndef FEATURE_TRACKER_HPP_
#define FEATURE_TRACKER_HPP_

#include "feature_struct.hpp"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"

#include "feature_extractor.hpp"
#include "descriptor_matcher.hpp"
#include "descriptor_head.hpp"
#include "dense_extractor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <Eigen/Dense>

#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

#include <rclcpp/logging.hpp>

#include "utility/metrics_logger.hpp"
#include "utility/debug_visualizer.hpp"

constexpr float OPTFLOW_DISTANCE_THRESHOLD_SQ = 0.25f;
constexpr float STEREO_FLOWBACK_THRESHOLD_SQ = 0.25f;

// other common utils
inline double distance_sq(const cv::Point2f &p1, const cv::Point2f &p2)
{
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

template <typename... Args>
static void reduceVectors(const std::vector<unsigned char> &status, Args &...vectors)
{
    const size_t N = status.size();
    if (((vectors.size() != N) || ...))
        return;

    size_t j = 0;
    for (size_t i = 0; i < N; ++i)
    {
        if (status[i])
        {
            if (j != i)
            {
                ((vectors[j] = std::move(vectors[i])), ...);
            }
            ++j;
        }
    }

    ((vectors.resize(j)), ...);
}

class FeatureTracker
{
public:
    enum class FeatureExtractionMethod
    {
        GFTT_CPU = 0,   // cv::goodFeaturesToTrack
        GFTT_CUDA = 1,  // cv::cuda::CornersDetector
        ALIKED = 2,     // ALIKED dense extractor + custom CUDA DKD + dhead engine
        SuperPoint = 3, // TensorRT SuperPoint
        RACO = 4,       // TensorRT RaCo (uses aliked dhead for descriptors when matched)
        XFeat = 5,      // TensorRT XFeat
        SIFT_CPU = 6,   // TODO: add cuda version of SIFT from https://github.com/alicevision/popsift
        DISK = 7,       // TODO: add Disk extractor
    };

    struct ExtractorProfile
    {
        std::string model_name;
        int input_channels = 0;
        bool is_dl = false;
    };

    enum class OpticalFlowBackend
    {
        CPU = 0,  // cv::calcOpticalFlowPyrLK (existing)
        CUDA = 1, // cv::cuda::SparsePyrLKOpticalFlow
    };

    // Use shared metrics type
    using FrameMetrics = uosm::utility::FeatureTrackerMetrics;

    struct FeatureTrackerParams
    {
        bool allow_flowback = true;
        bool enable_visualization = false;
        int row = 0;
        int col = 0;
        std::string calib_file_cam0;
        std::string calib_file_cam1;
        int min_feature_distance = 10U;
        int optflow_max_iterations = 30U;
        int max_tracked_keypoints = 512U;
        int optflow_pyramid_levels = 3U;
        int optflow_window_dim = 21U;
        float optflow_pyramid_scale = 0.5f;
        float optflow_epsilon = 0.01f;
        FeatureExtractionMethod feature_extraction_method = FeatureExtractionMethod::GFTT_CPU;
        OpticalFlowBackend optflow_device = OpticalFlowBackend::CPU;
        std::string weights_folder;
        bool use_descriptor_matcher = false;
        std::string matcher_type = "lightglue";
        float matcher_score_threshold = 0.0f;
        bool profile_trt_inference = false;
        // Stereo-only (ignored for mono).
        int stereo_window_dim = 21U;
        int stereo_pyramid_levels = 3U;
        float raco_cov_alpha = 1.0f;
        float raco_cov_floor_px = 0.0f;
        // Static FOV mask (fisheye/omnidirectional). Mono only.
        bool fisheye = false;
        std::string fisheye_mask;
    };

    explicit FeatureTracker(const FeatureTrackerParams &params);
    ~FeatureTracker();

    ObservationsMap trackImage(double _cur_time,
                               const cv::Mat &_img0,
                               const cv::Mat &_img1);

    cv::Mat getTrackVisualization();

    float getFocalLength() const { return focal_length_; }
    void setStereoBaseline(float baseline);

    void setStereoExtrinsics(const Eigen::Matrix3d &R_cam0_cam1,
                             const Eigen::Vector3d &t_cam0_cam1);

    const cv::Mat &getCurrentDescriptors() const { return current_descriptors_; }
    const std::vector<int> &getDescriptorIds() const { return descriptor_ids_; }
    int getDescriptorDim() const;

protected:
    bool enable_visualization_ = false;

private:
    cv::Mat getDescriptorDebugViz();

    bool allow_flowback_;
    int optflow_max_iterations_;
    int optflow_pyramid_levels_;
    int optflow_window_dim_;
    float optflow_pyramid_scale_;
    float optflow_epsilon_;

    int row_, col_;
    int n_id_, max_tracked_keypoints_;
    int min_feature_distance_;
    double prev_time_, cur_time_;
    cv::Mat prev_img_, cur_img0_, cur_img1_;
    cv::Mat mask_, tracked_image_;
    cv::Mat base_mask_;
    Point2fVec n_pts_;
    Point2fVec prev_pts0_, cur_pts0_, cur_pts1_;
    Point2fVec cur_undistorted_pts0_, cur_undistorted_pts1_;
    Point2fVec pts_vel0_, pts_vel1_;
    std::vector<int> ids0_, ids1_;
    std::vector<int> track_cnt_, track_cnt1_;

    Point2fMap prev_undistorted_pts0_map_, prev_undistorted_pts1_map_;
    Point2fMap prev_pts0_map;
    Point2fMap prev_pts0_viz_;

    FeatureExtractionMethod feature_extraction_method_;
    OpticalFlowBackend optflow_device_;
    std::string weights_folder_;
    bool use_descriptor_matcher_;
    std::string matcher_type_;
    float matcher_score_threshold_;
    bool use_stereo_mode_ = false;
    bool profile_trt_inference_;

    // Metrics logging (uses shared MetricsLogger)
    FrameMetrics current_metrics_;
    int frame_counter_ = 0;
    uosm::utility::MetricsLogger &metrics_logger_;

    bool is_cuda_optflow_init_ = false;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> cuda_lk_optflow_;
    cv::cuda::GpuMat prev_gpu_img_, cur_gpu_img0_, cur_gpu_img1_;
    cv::cuda::GpuMat prev_gpu_pts_, cur_gpu_pts0_, cur_gpu_pts1_;
    cv::cuda::GpuMat gpu_status_, gpu_error_;
    cv::Ptr<cv::cuda::CornersDetector> cuda_gftt_detector_;

    // GPU memory management
    cv::cuda::GpuMat gpu_new_pts_;
    cv::cuda::GpuMat gpu_mask_;
    bool frame_images_uploaded_ = false;
    bool mask_dirty_ = true; // Track whether GPU mask needs re-upload

    cv::cuda::HostMem pinned_img_buffer0_;
    cv::cuda::HostMem pinned_img_buffer1_;
    cv::cuda::HostMem host_new_pts_buffer_;
    bool pinned_buffers_initialized_ = false;

    // Stereo optical flow
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> cuda_lk_stereo_;
    int stereo_window_dim_;
    int stereo_pyramid_levels_;
    float stereo_baseline_;
    float focal_length_;

    Eigen::Matrix3d R_cam0_cam1_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cam0_cam1_ = Eigen::Vector3d::Zero();
    bool stereo_extrinsics_valid_ = false;

    std::array<camodocal::CameraPtr, 2> m_cameras_; // Stereo
    std::unique_ptr<uosm::perception::FeatureExtractor> feature_extractor_;
    std::unique_ptr<uosm::perception::DescriptorMatcher> descriptor_matcher_;
    std::unique_ptr<uosm::perception::DenseExtractor> dense_extractor_;

    std::vector<uosm::perception::DescriptorMatch> pending_stereo_matches_;
    bool stereo_matches_pending_ = false;
    std::unique_ptr<uosm::perception::DescriptorHead> desc_head_;
    cudaStream_t dl_chain_stream_ = nullptr;
    void *d_dkd_workspace_ = nullptr;
    size_t d_dkd_workspace_bytes_ = 0;
    float *d_dkd_kpts_px_ = nullptr; // (B, K, 2) — pixel coords from CUDA DKD
    float *d_dkd_kpts_n_ = nullptr;  // (B, K, 2) — normalised, fed into dhead
    float *d_dkd_scores_ = nullptr;  // (B, K)
    int *d_dkd_num_kpts_ = nullptr;  // (B,)
    // Pinned host mirror for keypoints + scores + counts (D2H once per frame).
    cv::cuda::HostMem dkd_pinned_host_;
    bool dkd_buffers_ready_ = false;

    // Per-frame descriptor cache for VPR / loop closure
    cv::Mat current_descriptors_;      // N x D float32, empty for GFTT modes
    std::vector<int> descriptor_ids_;  // feature IDs corresponding to descriptor rows
    std::vector<int> dl_new_left_ids_; // for loopclosure

    // Cached current-frame DL extractor outputs (populated in featureExtractDL on every frame in stereo mode)
    Point2fVec cur_dl_kps0_, cur_dl_kps1_; // left/right keypoint pixels
    cv::Mat cur_dl_desc0_;                 // lazy host mirror; empty until ensureHostMirrorDesc() is called
    int cur_dl_desc_dim_ = 0;
    bool cur_dl_valid_ = false;  // true if extractor produced data this frame
    bool cur_dl_stereo_ = false; // true if this frame ran a batch=2 inference

    void *cur_dl_d_desc0_ = nullptr;
    void *cur_dl_d_desc1_ = nullptr;
    int cur_dl_n0_ = 0;
    int cur_dl_n1_ = 0;
    bool cur_dl_host_desc0_valid_ = false;

    // Previous-frame DL cache
    Point2fVec prev_kf_left_kps_;
    cv::Mat prev_kf_left_desc_;
    std::vector<int> prev_kf_left_ids_;
    std::vector<int> prev_kf_left_track_cnt_;
    bool prev_kf_valid_ = false;

    // Per-stereo-match outcome for the debug overlay.
    struct StereoMatchViz
    {
        int idx_L = -1;
        int idx_R = -1;
        int outcome = 0;
    };
    std::vector<StereoMatchViz> stereo_match_viz_;

    float *d_prev_kf_desc_ = nullptr;
    size_t d_prev_kf_desc_capacity_bytes_ = 0;

    float *d_prev_kf_kps_ = nullptr;
    size_t d_prev_kf_kps_capacity_bytes_ = 0;

    // Direct DL-left-keypoint-index → cur_pts0_ slot (right side has no temporal track).
    std::vector<int> dl_to_pts0_idx_;

    // RaCo learned covariance
    std::vector<Eigen::Matrix2d> cur_dl_cov0_;
    std::unordered_map<int, Eigen::Matrix2d> cur_match_cov_;
    float raco_cov_alpha_ = 1.0f;
    float raco_cov_floor_px_ = 0.0f;

    // Persistent device-side scratch for the kept_rows indices passed to the
    // gather kernel in updateKeyframeCache(). Sized once for
    // max_tracked_keypoints_; reused every frame (left + right). Replaces a
    // per-frame cudaMalloc/cudaFree + sync H->D pattern.
    int *d_idx_persistent_ = nullptr;
    size_t d_idx_persistent_capacity_ints_ = 0;
    bool ensureIdxPersistent(size_t need_ints);

    // Encoder buffers (raco_aliked path).
    float *d_enc_kpts_n_ = nullptr; // (B=2, K, 2) float32, normalized [-1,1]
    size_t d_enc_kpts_n_capacity_bytes_ = 0;
    float *d_enc_image_b2_ = nullptr; // mono fallback: duplicates b=1 image into b=2
    size_t d_enc_image_b2_capacity_bytes_ = 0;
    float *d_prev_raco_kps_gpu_ = nullptr;
    int n_prev_raco_kps_ = 0;
    size_t d_prev_raco_kps_capacity_bytes_ = 0;
    std::unordered_map<int, int> id_alias_;

    // SIFT GPU buffers: (2 * max_kp * 128) descs, (2 * max_kp * 4) kps (xy+scale+ori)
    float *d_sift_desc_ = nullptr;
    size_t d_sift_desc_cap_ = 0;
    float *d_sift_kps_ = nullptr;
    size_t d_sift_kps_cap_ = 0;
    cv::Ptr<cv::SIFT> sift_detector_;

    inline bool inBorder(const cv::Point2f &pt) const;
    inline bool isMasked(const cv::Point2f &pt) const;
    inline bool readIntrinsicParameter(const std::string &calib_file, camodocal::CameraPtr &camera);
    void setMask();
    void drawTrack();
    Point2fVec undistortPoints(Point2fVec &pts, camodocal::CameraPtr &cam);
    Point2fVec calcVelfromPoints(const std::vector<int> &ids, const Point2fVec &pts,
                                 Point2fMap &prev_id_pts);
    void featureExtractOpenCV();

    // Temporal + stereo LK tracking (CUDA or CPU, selected by optflow_device_)
    void trackTemporalLK();
    void trackStereoLK();

    // CUDA optical flow internals
    void initCUDAOptflow();
    void cleanupCUDAOptflow();
    void trackTemporalCUDA();

    void featureExtractCUDA();   // GFTT on GPU
    void featureExtractDL();     // SuperPoint, RACO, XFeat via TensorRT
    void featureExtractAliked(); // ALIKED dense + custom CUDA DKD + dhead pipeline
    void featureExtractSIFT();   // OpenCV SIFT; uploads kps+descs to GPU for LightGlue path

    int kptInputDim() const
    {
        return feature_extraction_method_ == FeatureExtractionMethod::SIFT_CPU ? 4 : 2;
    }

    const float *currentKeypointsGpuBatch(int batch_idx) const;
    void runDescHeadHook();
    bool initDenseAlikedComponents(int batch, bool need_dhead);

    void recordExtractionTime(std::chrono::high_resolution_clock::time_point start);
    std::string resolveEnginePath(const std::string &role,
                                  const std::vector<std::string> &required_substrings,
                                  int max_kp,
                                  const std::vector<std::string> &forbidden_substrings = {},
                                  const std::string &base_dir_override = "") const;
    static ExtractorProfile profileFor(FeatureExtractionMethod method);

    void applyTemporalMatches(const std::vector<uosm::perception::DescriptorMatch> &matches,
                              const std::vector<int> &prev_ids,
                              const std::vector<int> &prev_track_cnt,
                              int n_cur,
                              std::vector<int> &cur_assigned_id,
                              std::vector<int> &cur_track_cnt,
                              std::unordered_set<int> &consumed_prev) const;
    // Left-cam only: the right cam carries no temporal track (per-frame stereo anchor).
    int accumulateLeftOutputs(const std::vector<cv::Point2f> &cur_dl_kps,
                              const std::vector<int> &cur_assigned_id,
                              const std::vector<int> &cur_track_cnt);
    void seedLeftFromCurrent(const std::vector<cv::Point2f> &cur_dl_kps);
    // Same-camera (temporal) essential-matrix RANSAC outlier filter.
    int filterMatchesE(std::vector<uosm::perception::DescriptorMatch> &matches,
                       const Point2fVec &prev_kps,
                       const Point2fVec &cur_kps,
                       const camodocal::CameraPtr &cam) const;
    // Cross-camera (stereo) essential-matrix RANSAC: idx0 keypoints use cam0, idx1 use cam1.
    int filterMatchesE(std::vector<uosm::perception::DescriptorMatch> &matches,
                       const Point2fVec &kps0,
                       const Point2fVec &kps1,
                       const camodocal::CameraPtr &cam0,
                       const camodocal::CameraPtr &cam1) const;

    void trackTemporalDescriptorLeft();
    void matchTemporalDescriptors();

    bool dlDescriptorTemporalActive() const;
    void buildStereoAnchors();

    // Reason a stereo descriptor match was rejected
    enum StereoRejectReason : uint8_t
    {
        Ok = 0,
        Depth = 1u << 0,    // 1: disparity below gate, depth outside window, or point behind a camera
        Reproj = 1u << 1,   // 2: inconsistent triangulation / reprojection residual too large
    };

    uint8_t acceptStereoOrphan(const cv::Point2f &l_und,
                               const cv::Point2f &r_und,
                               Eigen::Vector3d &P_cam0_out) const;

    // Lazy D2H of the current frame's DL descriptors. which=0 for left cam, 1 for right.
    // Called only when host-side access is needed (keyframe cache, VPR publishing).
    void ensureHostMirrorDesc();

    void updateKeyframeCache();
    void uploadFrameImages();
};
#endif // FEATURE_TRACKER_HPP_