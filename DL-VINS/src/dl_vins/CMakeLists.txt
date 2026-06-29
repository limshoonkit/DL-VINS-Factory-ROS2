cmake_minimum_required(VERSION 3.22.1)
project(dl_vins VERSION 1.0.0 LANGUAGES CXX C CUDA)

# ----------------------------
# Language standards / flags
# ----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86;87;89;")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unknown-pragmas -Xcompiler -Wno-cpp")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(DL_VINS_THIRD_PARTY_INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/../../Third_Party/install")
if(EXISTS "${DL_VINS_THIRD_PARTY_INSTALL}/opencv/lib/cmake/opencv4/OpenCVConfig.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${DL_VINS_THIRD_PARTY_INSTALL}/opencv")
    set(OpenCV_DIR "${DL_VINS_THIRD_PARTY_INSTALL}/opencv/lib/cmake/opencv4"
        CACHE PATH "OpenCV cmake config dir" FORCE)
endif()
if(EXISTS "${DL_VINS_THIRD_PARTY_INSTALL}/ceres/lib/cmake/Ceres/CeresConfig.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${DL_VINS_THIRD_PARTY_INSTALL}/ceres")
    set(Ceres_DIR "${DL_VINS_THIRD_PARTY_INSTALL}/ceres/lib/cmake/Ceres"
        CACHE PATH "Ceres cmake config dir" FORCE)
endif()

# ----------------------------
# ROS2 / build helpers
# ----------------------------
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(CUDAToolkit REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# ----------------------------
# Generate custom messages
# ----------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FeatureObservations.msg"
  "msg/FrameDescriptors.msg"
  DEPENDENCIES std_msgs
)

# ----------------------------
# Eigen, OpenCV, Ceres
# ----------------------------
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION} (${OpenCV_DIR})")
message(STATUS "Ceres version: ${Ceres_VERSION} (${Ceres_DIR})")

# ----------------------------
# TensorRT setup
# ----------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
    set(TENSORRT_LIB_DIR     "/usr/lib/aarch64-linux-gnu")
    set(TENSORRT_ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(TENSORRT_ROOT "$ENV{HOME}/Third_party/TensorRT-10.3.0.26"
        CACHE PATH "TensorRT install root (x86_64 tarball)")
    set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
    set(TENSORRT_LIB_DIR     "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
    set(TENSORRT_ARCH "x86_64")
else()
    message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Find TensorRT libraries explicitly
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_LIB_DIR}
    PATH_SUFFIXES lib
)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_LIB_DIR}
    PATH_SUFFIXES lib
)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS ${TENSORRT_LIB_DIR}
    PATH_SUFFIXES lib
)

find_library(TENSORRT_LIBRARY_PARSERS nvparsers
    HINTS ${TENSORRT_LIB_DIR}
    PATH_SUFFIXES lib
)

# Check if required libraries are found
if(NOT TENSORRT_LIBRARY_INFER)
    message(FATAL_ERROR "Could not find nvinfer library in ${TENSORRT_LIB_DIR}")
endif()

if(NOT TENSORRT_LIBRARY_INFER_PLUGIN)
    message(FATAL_ERROR "Could not find nvinfer_plugin library in ${TENSORRT_LIB_DIR}")
endif()

# Set up TensorRT libraries list
set(TENSORRT_LIBRARIES
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_INFER_PLUGIN}
)

# Add optional libraries if found
if(TENSORRT_LIBRARY_ONNXPARSER)
    list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_ONNXPARSER})
    message(STATUS "Added ONNX parser: ${TENSORRT_LIBRARY_ONNXPARSER}")
endif()

if(TENSORRT_LIBRARY_PARSERS)
    list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_PARSERS})
    message(STATUS "Added parsers: ${TENSORRT_LIBRARY_PARSERS}")
endif()

message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT libraries: ${TENSORRT_LIBRARIES}")

# Check TensorRT version
if(EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
    file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_VERSION_FILE)
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR ([0-9]+)" _ ${TENSORRT_VERSION_FILE})
    set(TENSORRT_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define NV_TENSORRT_MINOR ([0-9]+)" _ ${TENSORRT_VERSION_FILE})
    set(TENSORRT_MINOR ${CMAKE_MATCH_1})
    message(STATUS "Found TensorRT version: ${TENSORRT_MAJOR}.${TENSORRT_MINOR}")
endif()

# ----------------------------
# CUDA nvtx3 (header-only)
# ----------------------------
if(NOT TARGET CUDA::nvtx3)
  add_library(CUDA::nvtx3 INTERFACE IMPORTED)
  target_include_directories(CUDA::nvtx3 SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
  target_link_libraries(CUDA::nvtx3 INTERFACE ${CMAKE_DL_LIBS})
endif()

# ----------------------------
# Add library
# ----------------------------
add_library(feature_tracker_component SHARED
  src/feature_tracker_component.cpp
  src/feature/type_conv_helper.cu
  src/feature/feature_manager.cpp
  src/feature/feature_extractor.cpp
  src/feature/dense_extractor.cpp
  src/feature/descriptor_head.cpp
  src/feature/descriptor_matcher.cpp
  src/feature/lightglue_matcher.cpp
  src/feature/feature_tracker.cpp
  src/utility/debug_visualizer.cpp
)

# ----------------------------
# Dependencies and includes
# ----------------------------
ament_target_dependencies(feature_tracker_component
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  message_filters
  camera_models
)

target_include_directories(feature_tracker_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# ----------------------------
# Link libraries
# ----------------------------
target_link_libraries(feature_tracker_component
    ${TENSORRT_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${CUDAToolkit_LIBRARIES}
    CUDA::nvtx3
)

# ----------------------------
# Link generated message interfaces
# ----------------------------
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(feature_tracker_component ${cpp_typesupport_target})

# ----------------------------
# Set Runtime Paths
# ----------------------------
set_target_properties(feature_tracker_component PROPERTIES
    INSTALL_RPATH "${TENSORRT_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    LINK_FLAGS "-Wl,--enable-new-dtags"
)

# ----------------------------
# Register component
# ----------------------------
rclcpp_components_register_nodes(feature_tracker_component "uosm::perception::FeatureTrackerComponent")

# ----------------------------
# DL-VINS unified component (FeatureTracker + Estimator)
# ----------------------------
add_library(dl_vins_component SHARED
  src/dl_vins_component.cpp
  src/estimator/estimator.cpp
  src/initial/initial_sfm.cpp
  src/initial/initial_alignment.cpp
  src/initial/motion_estimator.cpp
  src/backend/ceres_backend.cpp
  src/backend/backend_factory.cpp
)

ament_target_dependencies(dl_vins_component
  rclcpp
  rclcpp_components
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  cv_bridge
  message_filters
  camera_models
  ament_index_cpp
)

target_include_directories(dl_vins_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(dl_vins_component
  feature_tracker_component
  ${CERES_LIBRARIES}
  ${OpenCV_LIBS}
  ${cpp_typesupport_target}
)

set_target_properties(dl_vins_component PROPERTIES
  INSTALL_RPATH "${TENSORRT_LIB_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
  LINK_FLAGS "-Wl,--enable-new-dtags"
)

rclcpp_components_register_nodes(dl_vins_component "uosm::perception::DlVinsComponent")

# ----------------------------
# Install rules
# ----------------------------
install(TARGETS feature_tracker_component dl_vins_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include 
        DESTINATION include)
install(DIRECTORY launch weights config
        DESTINATION share/${PROJECT_NAME}/)

# ----------------------------
# Export dependencies
# ----------------------------
ament_export_dependencies(
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  message_filters
  rosidl_default_runtime
)
ament_export_include_directories(include)
ament_export_libraries(feature_tracker_component dl_vins_component)
ament_package()