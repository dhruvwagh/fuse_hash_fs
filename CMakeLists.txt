cmake_minimum_required(VERSION 3.16)
project(LowLevelFuseHashFS CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE3 fuse3)

include_directories(${FUSE3_INCLUDE_DIRS})

add_executable(my_fs main.cpp)
target_link_libraries(my_fs ${FUSE3_LIBRARIES})
