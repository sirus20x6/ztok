cmake_minimum_required(VERSION 3.16)
project(ztok_hello C)

# Locate the installed ztok package. Pass `-DCMAKE_PREFIX_PATH=<install>`
# at configure time so this directory resolves to the install tree.
find_package(ztok REQUIRED)

add_executable(hello hello.c)
target_link_libraries(hello PRIVATE ztok::ztok)

add_executable(overlays overlays.c)
target_link_libraries(overlays PRIVATE ztok::ztok)

# Make the runtime loader find libztok.so next to where it was installed.
# Skip the RPATH dance entirely if your install lands in /usr/lib or
# another standard search path.
set_target_properties(hello overlays PROPERTIES
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "${ztok_DIR}/../.."
)
