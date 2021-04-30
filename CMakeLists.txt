cmake_minimum_required(VERSION 3.1.0)
project(pepper_surrogate)

# This assumes that you are using the provided binary files.
# To use newer versions of OpenHMD, simply compile your library and
# copy the binary and header into rviz_openhmd/OpenHMD in this repo. 
find_path (OpenHMD_INCLUDES openhmd.h
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/OpenHMD /usr/local/include /usr/include /sw/include
  )
find_library (OpenHMD_LIBRARIES libopenhmd.so
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/OpenHMD /usr/local/lib /usr/lib /lib /sw/lib
  )
include_directories(${OpenHMD_INCLUDES})
link_libraries(${OpenHMD_LIBRARIES})

link_directories(${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")

# Find and link the rviz library
find_package(catkin REQUIRED COMPONENTS
  rviz
  tf2_ros
)
catkin_package()
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

add_executable(head_control_node src/head_control_node.cpp)
target_link_libraries(head_control_node PRIVATE ${OpenHMD_LIBRARIES} ${catkin_LIBRARIES})

add_executable(simple src/simple.c)
target_link_libraries(head_control_node PRIVATE ${OpenHMD_LIBRARIES})

# Link all libraries into this one
# target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenHMD_LIBRARIES})