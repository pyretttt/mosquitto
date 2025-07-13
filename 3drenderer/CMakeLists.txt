cmake_minimum_required(VERSION 3.15)

project(3drenderer)


message(STATUS "Compilling with ${CMAKE_CXX_COMPILER_ID}")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

aux_source_directory(src SRC_SOURCES)

add_executable(${PROJECT_NAME} ${SRC_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE src)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -save-temps=obj)

add_subdirectory(3rd_party/SDL)

# UPNG
add_library(upng)
target_sources(upng PUBLIC 3rd_party/upng/upng.c)
configure_file(3rd_party/upng/upng.h 3rd_party/upng/upng.h COPYONLY)
target_include_directories(upng PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/3rd_party/upng)

target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2 upng)
