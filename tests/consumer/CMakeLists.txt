# Minimal CMake consumer used by tests/consumer/run.sh.
#
# Resolves the installed ztok via find_package(ztok) and links the tiny
# consumer.c against the imported ztok::ztok target. Pass
# `-DCMAKE_PREFIX_PATH=<install>` at configure time.
#
# This is intentionally separate from examples/c/CMakeLists.txt: the
# example links two demo programs that take a vocab-file argument; the
# test wants a no-input binary that returns 0 so run.sh can assert on the
# exit code alone.

cmake_minimum_required(VERSION 3.16)
project(ztok_consumer_test C)

find_package(ztok REQUIRED)

add_executable(consumer consumer.c)
target_link_libraries(consumer PRIVATE ztok::ztok)

# Embed an RPATH to the installed libdir so the loader finds libztok.so
# without LD_LIBRARY_PATH plumbing. ztok_DIR points at
# <prefix>/lib/cmake/ztok, so ../.. is <prefix>/lib.
set_target_properties(consumer PROPERTIES
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "${ztok_DIR}/../.."
    BUILD_RPATH "${ztok_DIR}/../.."
)
