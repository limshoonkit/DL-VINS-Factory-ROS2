#ifndef DESCRIPTOR_MATCHER_HPP_
#define DESCRIPTOR_MATCHER_HPP_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace uosm
{
    namespace perception
    {
        // Single descriptor match result.
        // idx0 / idx1 are indices into the kpts0 / kpts1 arrays that were passed to match().
        struct DescriptorMatch
        {
            int idx0;
            int idx1;
            float score;
        };

        // Abstract interface for any learned descriptor matcher
        //
        //  - d_kpts0_px / d_kpts1_px are device-side (n,2) float32 keypoints in original
        //    image pixel coordinates of size (image_h, image_w). Implementations handle
        //    their own model-space normalization on the GPU.
        //  - d_desc0/1 are device-side NxD float32 descriptors (row-major, already unit-norm).
        //    D is fixed by the engine and must match getDescriptorDim().
        //  - The returned matches are filtered by the implementation's score threshold.
        //  - match() MUST be thread-compatible with a single FeatureTracker thread.
        class DescriptorMatcher
        {
        public:
            virtual ~DescriptorMatcher() = default;

            virtual bool is_initialized() const = 0;
            virtual int getDescriptorDim() const = 0;
            virtual int getMaxKeypoints() const = 0;
            virtual const std::string &name() const = 0; // e.g. "lightglue", "omniglue", "loma", ...

            virtual std::vector<DescriptorMatch> match(const float *d_kpts0_px, int n0,
                                                          const void *d_desc0,
                                                          const float *d_kpts1_px, int n1,
                                                          const void *d_desc1,
                                                          int image_h, int image_w) = 0;

            // Bundle of inputs for paired (concurrent) matching.
            struct MatchInputs
            {
                const float *d_kpts0_px = nullptr;
                int n0 = 0;
                const void *d_desc0 = nullptr;
                const float *d_kpts1_px = nullptr;
                int n1 = 0;
                const void *d_desc1 = nullptr;
                int image_h = 0;
                int image_w = 0;
            };

            // Match several independent pairs in one call. Returns one match vector per
            // input pair, in the same order. Implementations may run a single batched
            // inference; the default falls back to a per-pair match loop. The pairs
            // MUST share no overlapping device buffers, since they may all be in flight
            // at once.
            virtual std::vector<std::vector<DescriptorMatch>>
            matchBatched(const std::vector<MatchInputs> &pairs)
            {
                std::vector<std::vector<DescriptorMatch>> out;
                out.reserve(pairs.size());
                for (const auto &p : pairs)
                    out.push_back(match(p.d_kpts0_px, p.n0, p.d_desc0,
                                           p.d_kpts1_px, p.n1, p.d_desc1,
                                           p.image_h, p.image_w));
                return out;
            }
        };

        struct DescriptorMatcherParams
        {
            std::string engine_path;
            int descriptor_dim = 256;
            int max_keypoints = 256;
            float score_threshold = 0.5f;
        };

        // Factory: construct a matcher by type string ("lightglue", "omniglue", "loma", ...).
        // Unknown types return nullptr and log a warning at the call site.
        std::unique_ptr<DescriptorMatcher> makeDescriptorMatcher(const std::string &type,
                                                                 const DescriptorMatcherParams &params);

    } // namespace perception
} // namespace uosm

#endif // DESCRIPTOR_MATCHER_HPP_
