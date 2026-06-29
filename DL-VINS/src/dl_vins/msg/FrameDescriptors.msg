# Per-frame learned feature descriptors for VPR / loop closure
# Published only when using DL frontends (ALIKED, SuperPoint)

std_msgs/Header header

# Feature IDs matching newly extracted keypoints
uint32[] feature_ids

# Descriptor dimension (e.g. 128 for ALIKED, 256 for SuperPoint)
uint32 descriptor_dim

# Flattened descriptor matrix (N x descriptor_dim, row-major)
float32[] descriptors
