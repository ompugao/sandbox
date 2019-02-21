mkdir -p build
cd build
pyenv shell sandbox
PREFIX_MAIN=`pyenv virtualenv-prefix`
PREFIX=`pyenv prefix`
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX="$PREFIX" \
    -D PYTHON_EXECUTABLE="$PREFIX"/bin/python3.7 \
    -D PYTHON_PACKAGES_PATH="$PREFIX"/lib/python3.7/site-packages \
    -D PYTHON_LIBRARY="$PREFIX_MAIN"/lib/libpython3.7m.so \
    -D PYTHON_INCLUDE_DIR="$PREFIX_MAIN"/include/python3.7m \
    -D PYTHON_NUMPY_INCLUDE_DIRS="$PREFIX"/lib/python3.7/site-packages/numpy/core/include ..
make -C build
