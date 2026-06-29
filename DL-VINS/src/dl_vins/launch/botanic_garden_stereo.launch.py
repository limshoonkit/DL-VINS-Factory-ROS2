from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

import os

VALID_EXTRACTORS_STEREO = [
    "gftt_cpu",
    "gftt_gpu",
    "sift_cpu_lightglue",
    "aliked_lightglue",
    "superpoint_lightglue",
    "raco_lightglue",
    "xfeat_lightglue",
    "aliked_lk",
    "superpoint_lk",
    "raco_lk",
    "xfeat_lk",
    "sift_cpu_lk",
]

def _make_launch(context, *args, **kwargs):
    extractor = LaunchConfiguration("extractor").perform(context)
    if extractor not in VALID_EXTRACTORS_STEREO:
        raise RuntimeError(
            f"extractor='{extractor}' not in VALID_EXTRACTORS_STEREO {VALID_EXTRACTORS_STEREO}"
        )

    pkg_share = FindPackageShare("dl_vins").perform(context)
    config_yaml = os.path.join(pkg_share, "config", "botanic_garden", f"stereo_{extractor}.yaml")

    weights_override = {"weights_folder": "weights/botanic_garden"}
    _max_kp = LaunchConfiguration("max_keypoints").perform(context).strip()
    kp_override = {"max_tracked_keypoints": int(_max_kp)} if _max_kp else {}

    use_loop = LaunchConfiguration("use_loop_fusion").perform(context).lower() in (
        "true", "1", "yes",
    )

    use_sim_time_param = {"use_sim_time": True}

    use_rviz = LaunchConfiguration("rviz").perform(context).lower() in (
        "true", "1", "yes",
    )
    # When visualizing live, force the feature-track image to be published.
    use_foxglove = LaunchConfiguration("foxglove").perform(context).lower() in (
        "true", "1", "yes",
    )
    # Either visualizer needs the same scene: feature-track image + a fixed world frame.
    use_viz = use_rviz or use_foxglove
    viz_override = {"enable_visualization": True} if use_viz else {}

    # Botanic Garden: cam0 = left (Dalsa gray down), cam1 = right.
    composable_nodes = [
        ComposableNode(
            package="dl_vins",
            plugin="uosm::perception::DlVinsComponent",
            name="dl_vins_component",
            parameters=[config_yaml, weights_override, kp_override, use_sim_time_param, viz_override],
            remappings=[
                ("imu", "/imu/data"),
                ("image0", "/dalsa_gray/left/image_raw"),
                ("image1", "/dalsa_gray/right/image_raw"),
            ],
            extra_arguments=[{"use_intra_process_comms": True}],
        ),
    ]

    if use_loop:
        loop_yaml = os.path.join(
            FindPackageShare("loop_fusion").perform(context),
            "config",
            "loop_fusion_botanic_garden.yaml",
        )
        cam0_calib_path = os.path.join(
            pkg_share, "config", "botanic_garden", "dalsa_cam0_pinhole_down.yaml")
        if extractor.endswith("_lk") or extractor.startswith("gftt"):
            loop_mode = "classic"
        elif (extractor.startswith("aliked") or extractor.startswith("raco")
              or extractor.startswith("superpoint") or extractor.startswith("xfeat")
              or extractor.startswith("sift_cpu_lightglue")):
            loop_mode = "global_vpr"
        else:
            loop_mode = "classic"
        import yaml as _yaml
        try:
            with open(config_yaml) as _f:
                _dl_params = _yaml.safe_load(_f).get(
                    "dl_vins_component", {}
                ).get("ros__parameters", {})
            _enable_csv = bool(_dl_params.get("enable_csv_logging", False))
            _log_folder = str(_dl_params.get("log_folder", "./tmp/dl_vins_logs"))
        except Exception:
            _enable_csv = False
            _log_folder = "./tmp/dl_vins_logs"
        composable_nodes.append(
            ComposableNode(
                package="loop_fusion",
                plugin="uosm::perception::LoopFusionComponent",
                name="loop_fusion",
                parameters=[{
                    "config_file": loop_yaml,
                    "cam0_calib_path": cam0_calib_path,
                    "loop_closure_mode": loop_mode,
                    "descriptor_topic": "/dl_vins/frame_descriptors",
                    "enable_csv_logging": _enable_csv,
                    "log_folder": _log_folder,
                    "use_sim_time": True,
                }],
                remappings=[
                    ("/odometry", "/dl_vins/odometry"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            )
        )

    container = ComposableNodeContainer(
        name="dl_vins_container",
        namespace="dl_vins",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=composable_nodes,
        output="screen",
    )
    nodes = [container]

    if use_viz:
        # Identity world->odom so the VIO `odom` layer and the loop_fusion/PGO
        # `world` layer coexist under one fixed frame in RViz/Foxglove.
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="world_to_odom_static_tf",
                arguments=["0", "0", "0", "0", "0", "0", "world", "odom"],
                parameters=[use_sim_time_param],
                output="screen",
            )
        )

    if use_rviz:
        rviz_config = os.path.join(pkg_share, "config", "dl_vins_stereo.rviz")
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                parameters=[use_sim_time_param],
                output="screen",
            )
        )

    if use_foxglove:
        # Foxglove Studio (app or studio.foxglove.dev) connects to ws://<host>:8765.
        nodes.append(
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                parameters=[use_sim_time_param, {"port": 8765}],
                output="screen",
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "extractor",
            default_value="aliked_lightglue",
            description=f"One of {VALID_EXTRACTORS_STEREO}",
        ),
        DeclareLaunchArgument(
            "max_keypoints",
            default_value="",
            description="Override max_tracked_keypoints for the keypoint ablation",
        ),
        DeclareLaunchArgument(
            "use_loop_fusion",
            default_value="false",
            description="Compose the loop_fusion node in the same container",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="false",
            description="Start RViz2 + static world->odom TF and force "
                        "enable_visualization for the feature-track image",
        ),
        DeclareLaunchArgument(
            "foxglove",
            default_value="false",
            description="Start foxglove_bridge (ws://:8765) + static world->odom "
                        "TF and force enable_visualization for the feature-track image",
        ),
        OpaqueFunction(function=_make_launch),
    ])
