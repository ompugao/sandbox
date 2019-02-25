
all: build_debug build_release


.PHONY: build_debug
build_debug:
	mkdir -p build_dbg
	cd build_dbg && cmake ../ -DCMAKE_BUILD_TYPE=Debug && make -j4

.PHONY: build_release
build_release:
	mkdir -p build
	cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release && make -j4

.PHONY: clean
clean:
	rm build build_dbg -r
