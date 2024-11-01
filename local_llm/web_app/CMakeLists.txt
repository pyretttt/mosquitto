cmake_minimum_required(VERSION 3.23)

project(web_app)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

aux_source_directory(src SRC_SOURCES)

add_executable(${PROJECT_NAME} ${SRC_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE src)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -save-temps=obj)
