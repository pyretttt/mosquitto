cmake_minimum_required(VERSION 3.23)

project(opengl_playground)

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 7b6aead9fb88b3623e3b3725ebb42670cbe4c579
)
FetchContent_MakeAvailable(glfw)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_SOURCES)

add_executable(opengl_playground ${SRC_SOURCES})
target_link_libraries(opengl_playground PRIVATE glfw)
