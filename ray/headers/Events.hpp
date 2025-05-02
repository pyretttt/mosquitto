#pragma once

#include <variant>

#include "SDL.h"

using Event = std::variant<SDL_Event>;