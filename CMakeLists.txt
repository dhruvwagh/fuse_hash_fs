cmake_minimum_required(VERSION 3.16)
project(LowLevelFuseHashFS CXX)

set(CMAKE_CXX_STANDARD 17)

# Make FUSE3 required and add pthread
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE3 REQUIRED fuse3)  # Added REQUIRED
find_package(Threads REQUIRED)  # Add pthread support

# Include directories
include_directories(${FUSE3_INCLUDE_DIRS})

# Add executable and link libraries
add_executable(my_fs main.cpp)
target_link_libraries(my_fs 
    ${FUSE3_LIBRARIES}
    Threads::Threads  # Add pthread linking
)