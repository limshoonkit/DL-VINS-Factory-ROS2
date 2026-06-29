# Feature observations from the feature tracker
# Used to communicate tracked features to the estimator

std_msgs/Header header

# Feature IDs (unique per feature track)
uint32[] feature_ids

# Number of features (for validation)
uint32 num_features

# Left camera observations (normalized coordinates)
float64[] points_left_x
float64[] points_left_y

# Right camera observations (normalized coordinates)
# Empty or zero if no stereo match for that feature
float64[] points_right_x
float64[] points_right_y

# Pixel coordinates (for visualization and debugging)
float64[] pixel_left_x
float64[] pixel_left_y
float64[] pixel_right_x
float64[] pixel_right_y

# Feature velocities (normalized coordinates per second)
float64[] velocities_x
float64[] velocities_y
