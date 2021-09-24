#add opencv
cmake_minimum_required(VERSION 3.10)
project( lumina )
find_package(OpenCV 4.5.1 REQUIRED)

add_definitions(-std=c++11)

set(LUMINA_SOURCES
    model/leNet.cpp
    main.cpp
    )


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/Math)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/Array)

add_executable( lumina main.cpp ${LUMINA_SOURCES} )
target_link_libraries( lumina ${OpenCV_LIBS} )