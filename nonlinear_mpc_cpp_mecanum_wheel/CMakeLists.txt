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

find_package(PythonInterp)
find_package(PythonLibs REQUIRED)
if(NOT PYTHON_NUMPY_INCLUDE_DIRS)
  # Attempt to discover the NumPy include directory. If this succeeds, then build python API with NumPy
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c
                    "import os; os.environ['DISTUTILS_USE_SDK']='1'; import numpy.distutils; print(os.pathsep.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
                  RESULT_VARIABLE PYTHON_NUMPY_PROCESS
                  OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIRS
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(PYTHON_NUMPY_PROCESS EQUAL 0)
    file(TO_CMAKE_PATH "${PYTHON_NUMPY_INCLUDE_DIRS}" _PYTHON_NUMPY_INCLUDE_DIRS)
    set(PYTHON_NUMPY_INCLUDE_DIRS "${_PYTHON_NUMPY_INCLUDE_DIRS}" CACHE PATH "Path to numpy headers")
  endif()
endif()

if(PYTHON_NUMPY_INCLUDE_DIRS)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.version.version)"
                  OUTPUT_VARIABLE PYTHON_NUMPY_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

include_directories(${PYTHON_INCLUDE_PATH} ${PYTHON_NUMPY_INCLUDE_DIRS})

# main
add_executable(main main.cpp)
target_link_libraries(main ${CERES_LIBRARIES} ${PYTHON_LIBRARIES} gflags stdc++fs)
