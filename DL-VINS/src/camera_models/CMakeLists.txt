cmake_minimum_required(VERSION 3.22.1)
project(camera_models)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
find_package(std_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(camera_models STATIC
  src/chessboard/Chessboard.cc
  src/calib/CameraCalibration.cc
  src/camera_models/Camera.cc
  src/camera_models/CameraFactory.cc
  src/camera_models/CostFunctionFactory.cc
  src/camera_models/PinholeCamera.cc
  src/camera_models/PinholeFullCamera.cc
  src/camera_models/CataCamera.cc
  src/camera_models/EquidistantCamera.cc
  src/camera_models/ScaramuzzaCamera.cc
  src/sparse_graph/Transform.cc
  src/gpl/gpl.cc
)

target_link_libraries(camera_models
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
)

ament_target_dependencies(camera_models
  rclcpp
  std_msgs
)

add_executable(Calibrations
  src/intrinsic_calib.cc
)

target_link_libraries(Calibrations
  camera_models
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
)

ament_target_dependencies(Calibrations
  rclcpp
  std_msgs
  OpenCV
)

install(TARGETS
  camera_models
  Calibrations
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_export_include_directories(include)
ament_export_libraries(camera_models)
ament_export_dependencies(std_msgs rclcpp OpenCV Ceres)
ament_package()
