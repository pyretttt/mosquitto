cmake_minimum_required(VERSION 3.2)

project(3drenderer)


message(STATUS "Compilling with ${CMAKE_CXX_COMPILER_ID}")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

aux_source_directory(src src_sources)

add_executable(${PROJECT_NAME} ${src_sources})

target_include_directories(${PROJECT_NAME} PRIVATE src)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -save-temps=obj)

add_subdirectory(3rd_party/SDL)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)