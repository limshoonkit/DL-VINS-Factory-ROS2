cmake_minimum_required(VERSION 3.22.1)
project(loop_fusion VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86;87;89;")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unknown-pragmas -Xcompiler -Wno-cpp")
set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
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

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_models REQUIRED)
find_package(dl_vins REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

message(STATUS "loop_fusion OpenCV: ${OpenCV_VERSION} (${OpenCV_DIR})")
message(STATUS "loop_fusion Ceres:  ${Ceres_VERSION} (${Ceres_DIR})")

# ----------------------------
# TensorRT setup
# ----------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
  set(TENSORRT_LIB_DIR     "/usr/lib/aarch64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(TENSORRT_ROOT "$ENV{HOME}/Third_party/TensorRT-10.3.0.26"
      CACHE PATH "TensorRT install root (x86_64 tarball)")
  set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
  set(TENSORRT_LIB_DIR     "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
else()
  message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_library(TENSORRT_LIBRARY_INFER nvinfer HINTS ${TENSORRT_LIB_DIR})
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin HINTS ${TENSORRT_LIB_DIR})
if(NOT TENSORRT_LIBRARY_INFER)
  message(FATAL_ERROR "loop_fusion: could not find nvinfer in ${TENSORRT_LIB_DIR}")
endif()
set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_INFER})
if(TENSORRT_LIBRARY_INFER_PLUGIN)
  list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_INFER_PLUGIN})
endif()
message(STATUS "loop_fusion TensorRT: ${TENSORRT_LIBRARIES}")

add_library(loop_fusion_component SHARED
  src/loop_fusion_component.cpp
  src/pose_graph.cpp
  src/keyframe.cpp
  src/vlad.cpp
  src/dino_vpr.cpp
  src/dino_vpr_preprocess.cu
  src/utility/camera_pose_visualization.cpp
  src/utility/utility.cpp
  src/ThirdParty/DBoW/BowVector.cpp
  src/ThirdParty/DBoW/FBrief.cpp
  src/ThirdParty/DBoW/FeatureVector.cpp
  src/ThirdParty/DBoW/QueryResults.cpp
  src/ThirdParty/DBoW/ScoringObject.cpp
  src/ThirdParty/DUtils/Random.cpp
  src/ThirdParty/DUtils/Timestamp.cpp
  src/ThirdParty/DVision/BRIEF.cpp
  src/ThirdParty/VocabularyBinary.cpp
)

target_include_directories(loop_fusion_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

set_target_properties(loop_fusion_component PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INSTALL_RPATH "${TENSORRT_LIB_DIR}")

ament_target_dependencies(loop_fusion_component
  rclcpp
  rclcpp_components
  ament_index_cpp
  std_msgs
  std_srvs
  sensor_msgs
  nav_msgs
  visualization_msgs
  geometry_msgs
  cv_bridge
  camera_models
  dl_vins
)

target_link_libraries(loop_fusion_component
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${TENSORRT_LIBRARIES}
  CUDA::cudart
)

rclcpp_components_register_nodes(loop_fusion_component
  "uosm::perception::LoopFusionComponent")

install(TARGETS loop_fusion_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY support_files config
  DESTINATION share/${PROJECT_NAME}/)

ament_export_dependencies(
  rclcpp
  rclcpp_components
  sensor_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  camera_models
)
ament_export_include_directories(include)
ament_export_libraries(loop_fusion_component)
ament_package()
