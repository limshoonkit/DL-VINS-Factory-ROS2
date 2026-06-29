cmake_minimum_required(VERSION 3.8...3.16)
project(brisk VERSION 2.0.7 LANGUAGES CXX)

option(BRISK_BUILD_DEMO "Whether to build the demo" ON)
option(BUILD_SHARED_LIBS "Whether to build brisk as a shared library" OFF)

set(BRISK_PRIVATE_CXX_FLAGS -Wall -Wextra -Wpedantic)

# Make sure we build in Release and warn otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT DEFINED ENV{CMAKE_BUILD_TYPE})
  set(CMAKE_BUILD_TYPE Release CACHE STRING "The type of build." FORCE)
  # Set the possible values of CMAKE_BUILD_TYPE for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(WARNING "brisk CMAKE_BUILD_TYPE not set to 'Release', performance may be terrible.")
endif()

# Prepare for installing libraries
include(GNUInstallDirs)
# The directory where the .cmake files will be installed
if(WIN32 AND NOT CYGWIN)
  set(INSTALL_CMAKEDIR "brisk/cmake")
elseif(APPLE)
  set(INSTALL_CMAKEDIR "brisk.framework/Resources/CMake")
else()
  set(INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/brisk")
endif()

# Build the agast library.
add_subdirectory(agast)

# Generate the configured header file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/briskConfig.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/include/brisk/briskConfig.hpp" @ONLY)

# Build the brisk library.
find_package(OpenCV REQUIRED COMPONENTS core features2d)
add_library(brisk
  src/brisk-descriptor-extractor.cc
  src/brisk-feature-detector.cc
  src/brisk-layer.cc
  src/brisk-scale-space.cc
  src/brute-force-matcher.cc
  src/harris-feature-detector.cc
  src/harris-score-calculator-float.cc
  src/harris-score-calculator.cc
  src/harris-scores.cc
  src/image-down-sampling.cc
  src/pattern-provider.cc
  src/vectorized-filters.cc
)
target_include_directories(brisk
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)
target_compile_features(brisk PUBLIC cxx_std_11)
target_link_libraries(brisk PUBLIC ${OpenCV_LIBS} brisk::agast)
target_compile_options(brisk PRIVATE ${BRISK_PRIVATE_CXX_FLAGS})
# Enable vector instructions.
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm*")
  message(STATUS "ARM processor detected, will attempt to use NEON.")
  target_compile_definitions(brisk PUBLIC -D__ARM_NEON__)
  target_compile_options(brisk PUBLIC -flax-vector-conversions)
else()
  message(STATUS "Assuming SSE instructions available.")
  target_compile_options(brisk PUBLIC -mssse3)
endif()
# Add an alias target so that brisk can be linked the same no matter whether
# find_package or add_subdirectory is used.
add_library(brisk::brisk ALIAS brisk)

# Generate *Config.cmake and *ConfigVersion.cmake files
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/briskConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/briskConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKEDIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/briskConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
# Install files
install(TARGETS brisk EXPORT briskTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/brisk DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
install(DIRECTORY include/sse2neon DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/brisk DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
install(EXPORT briskTargets
  FILE briskTargets.cmake
  NAMESPACE brisk::
  DESTINATION ${INSTALL_CMAKEDIR})
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/briskConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/briskConfigVersion.cmake
  DESTINATION ${INSTALL_CMAKEDIR})



if(BRISK_BUILD_DEMO)
  find_package(OpenCV REQUIRED COMPONENTS imgproc highgui)
  include(CheckIncludeFileCXX)
  set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
  check_include_file_cxx("opencv2/nonfree/nonfree.hpp" HAVE_OPENCV_NONFREE_HPP)
  if(HAVE_OPENCV_NONFREE_HPP)
    add_executable(demo src/demo.cc)
  else()
    message(WARNING "Non-free OpenCV components not available, demo application with limited functionality")
    add_executable(demo src/demo-free.cc)
  endif()
  target_compile_definitions(demo PRIVATE BRISK_IMAGE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/images\")
  target_compile_options(demo PRIVATE ${BRISK_PRIVATE_CXX_FLAGS})
  target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(demo ${OpenCV_LIBS} brisk::brisk)
endif()
