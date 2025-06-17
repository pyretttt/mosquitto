#pragma once

#include <vector>

namespace gl {
    using ID = unsigned int;

    // Required because by default all uniforms are initialized to 0. So unitialized samplers might have other first setted texture location
    constexpr size_t samplerLocationOffset = 16;
}