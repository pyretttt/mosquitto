#!/usr/bin/env sh
(cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug &&
cmake --build ./build -j 4 &&
cd build && 
./opengl_playground)