cmake_minimum_required(VERSION 3.15)

project(redis VERSION 1.0 LANGUAGES CXX)


add_definitions("-std=c++20")

set(WARNINGS -Wall -Wextra -Wpedantic)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/common COMMON_SOURCES)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/client CLIENT_SOURCES COMMON_SOURCES)
add_executable(redis_client ${CLIENT_SOURCES})
target_include_directories(redis_client
                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/client
                            ${CMAKE_CURRENT_SOURCE_DIR}/common
                            )
target_compile_options(redis_client PRIVATE ${WARNINGS})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/server SERVER_SOURCES COMMON_SOURCES)
add_executable(redis_server ${SERVER_SOURCES})
target_include_directories(redis_server 
                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/server
                            ${CMAKE_CURRENT_SOURCE_DIR}/common
                            )
target_compile_options(redis_server PRIVATE ${WARNINGS})
