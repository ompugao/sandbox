
all: build_debug build_release build_relwithdebinfo

CXX=clang++

.PHONY: build_debug
build_debug:
	mkdir -p build_dbg
	cd build_dbg && cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ && time make -j4

.PHONY: build_release
build_release:
	mkdir -p build
	cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ && time make -j4

.PHONY: build_relwithdebinfo
build_relwithdebinfo:
	mkdir -p build_rwdi
	cd build_rwdi && cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++ && time make -j4


.PHONY: clean
clean:
	rm build build_dbg build_rwdi -rf
