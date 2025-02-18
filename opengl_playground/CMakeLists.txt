cmake_minimum_required(VERSION 3.23)

project(opengl_playground VERSION 1.0.0 LANGUAGES CXX)

# Fetch GLFW
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 7b6aead9fb88b3623e3b3725ebb42670cbe4c579
)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0af55ccecd98d4e5a8d1fad7de25ba429d60e863
)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0af55ccecd98d4e5a8d1fad7de25ba429d60e863
)
FetchContent_MakeAvailable(glfw glm)

find_package(OpenGL REQUIRED)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_SOURCES)
add_executable(opengl_playground ${SRC_SOURCES})
target_compile_features(opengl_playground PRIVATE cxx_std_23)
target_link_libraries(opengl_playground PRIVATE glfw ${OPENGL_gl_LIBRARY} glm::glm)
target_include_directories(opengl_playground PRIVATE src ${GLFW_SOURCE_DIR}/deps ${OPENGL_INCLUDE_DIR}) # to include glad for opengl

target_link_libraries(opengl_playground PRIVATE "-framework Cocoa")
target_link_libraries(opengl_playground PRIVATE "-framework OpenGL")
target_link_libraries(opengl_playground PRIVATE "-framework IOKit")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/shader.fs
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/shader.vs
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/resources
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})