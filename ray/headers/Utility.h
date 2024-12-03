#pragma once

#include <functional>

template <typename T>
constexpr inline T &&modified(T &&object, std::function<void(T &)> modification) {
    modification(object);
    return object;
}