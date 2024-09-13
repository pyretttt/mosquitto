cmake_minimum_required(VERSION 3.15)

project(redis VERSION 1.0 LANGUAGES CXX)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SOURCES)

add_executable(redis ${SOURCES})

target_include_directories(redis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

