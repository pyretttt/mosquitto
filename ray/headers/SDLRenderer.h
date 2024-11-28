#include <utility>
#include <functional>

#include "SDL.h"

#include "Renderer.h"

struct SDLRenderer : public Renderer {
    SDLRenderer(SDL_Window *window);
    void update() const override;
    void render() const override;

    ~SDLRenderer();
private:
    std::pair<int, int> windowSize;
    std::allocator<uint32_t> allocator;
    SDL_Renderer *renderer;
    std::unique_ptr<uint32_t, std::function<void(uint32_t *p)>> colorBuffer;
    std::unique_ptr<uint32_t, std::function<void(uint32_t *p)>> zBuffer;
    SDL_Texture *renderTarget;
};