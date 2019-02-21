cmake_minimum_required(VERSION 2.8)

project(linear_mpc_cpp)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS ON) #...without compiler extensions like gnu++11

find_package(Threads REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIRECTORIES})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH} ${PYTHON_NUMPY_INCLUDE_DIRS})

# main
add_executable(main main.cpp)
target_link_libraries(main ${CERES_LIBRARIES} ${PYTHON_LIBRARIES} gflags)
