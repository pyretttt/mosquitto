#include "SDL.h"

#include "Renderer.h"

struct SDLRenderer : public Renderer {
    SDLRenderer() = default;
    void update();
    void render();

    ~SDLRenderer();
};