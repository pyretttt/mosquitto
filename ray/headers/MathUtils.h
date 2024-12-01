#pragma once

#include <type_traits>

#include "Eigen/Dense"

template <typename Vector>
inline constexpr std::decay<Vector>::type projection(Vector &&a, Vector &&on) {
    auto const onNorm = on.normalized();
    return (onNorm * a.dot(onNorm)).eval();
}

template <typename Vector>
inline constexpr std::decay<Vector>::type rejection(Vector &&a, Vector &&on) {
    return (a - projection(a, on)).eval();
}