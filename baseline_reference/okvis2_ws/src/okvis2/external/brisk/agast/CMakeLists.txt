cmake_minimum_required(VERSION 3.8...3.16)

find_package(OpenCV REQUIRED COMPONENTS core features2d)

add_library(agast
  src/agast5-8-nms.cc  
  src/agast5-8.cc  
  src/agast7-12d-nms.cc  
  src/agast7-12d.cc  
  src/agast7-12s-nms.cc  
  src/agast7-12s.cc  
  src/ast-detector.cc  
  src/non-maximum-suppression.cc  
  src/oast9-16-nms.cc
  src/oast9-16.cc  
)
target_include_directories(agast
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(agast PUBLIC ${OpenCV_LIBS})
target_compile_features(agast PUBLIC cxx_std_11)
add_library(brisk::agast ALIAS agast)

install(TARGETS agast EXPORT briskTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/agast DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
